"""
Phase 6 — Tournament Simulation & Strategy Layer
=================================================
Reads:  data/phase5_ensemble_probs.csv                — win probs for every matchup
        data/march-machine-learning-mania-2026/        — Kaggle bracket data
        data/ncaa_net.csv                              — NET rankings (seeding fallback)
Writes: data/phase6_team_round_probs.csv    — P(team reaches each round)
        data/phase6_upset_paths.csv         — key upset correlation structure
        data/phase6_brackets.json           — 3 strategy bracket picks
        data/phase6_pool_ev.csv             — expected pool points per bracket
        data/phase6_simulation_plots.png    — round reach + EV plots
        data/phase6_simulation_raw.csv      — sampled tournament outcomes (10k rows)

Usage:
    python phase6_simulation.py
    python phase6_simulation.py --sims 100000 --scoring 1,2,4,8,16,32
    python phase6_simulation.py --scoring 1,2,4,8,16,32 --upset-multiplier 2.0
"""

import argparse
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
ENSEMBLE_PROBS  = "data/phase5_ensemble_probs.csv"
KAGGLE_DIR      = "data/march-machine-learning-mania-2026"
NET_PATH        = "data/ncaa_net.csv"

ROUND_PROBS_OUT = "data/phase6_team_round_probs.csv"
UPSET_PATHS_OUT = "data/phase6_upset_paths.csv"
BRACKETS_OUT    = "data/phase6_brackets.json"
POOL_EV_OUT     = "data/phase6_pool_ev.csv"
SIM_PLOTS_OUT   = "data/phase6_simulation_plots.png"
SIM_RAW_OUT     = "data/phase6_simulation_raw.csv"

DEFAULT_SIMS            = 10_000
DEFAULT_SCORING         = [1, 2, 4, 8, 16, 32]  # standard ESPN-style
DEFAULT_UPSET_MULT      = 1.0   # set >1 to reward upsets

# Bracket structure: 4 regions × 16 teams, standard seeding matchups
REGIONS = ["W", "X", "Y", "Z"]
REGION_NAMES = {"W": "East", "X": "West", "Y": "South", "Z": "Midwest"}
# Round-of-64 seed matchups (strong_seed vs weak_seed within a region)
R1_MATCHUPS = [(1,16),(2,15),(3,14),(4,13),(5,12),(6,11),(7,10),(8,9)]

ROUND_NAMES = {
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Championship",
}


# ── Load win probability lookup ────────────────────────────────────────────────
def build_prob_lookup(ensemble_df):
    """
    Returns a function win_prob(team_a, team_b) → P(team_a beats team_b).
    Uses bidirectional lookup; defaults to 0.5 if matchup not found.
    """
    lookup = {}
    for _, row in ensemble_df.iterrows():
        a = str(row["team_a"]).strip()
        b = str(row["team_b"]).strip()
        p = float(row["ensemble_prob"])
        lookup[(a, b)] = p
        lookup[(b, a)] = 1 - p

    def win_prob(a, b):
        if a == b:
            return 0.5
        if (a, b) in lookup:
            return lookup[(a, b)]
        if (b, a) in lookup:
            return 1 - lookup[(b, a)]
        return 0.5  # unknown matchup → coin flip

    return win_prob


# ── Load / build 2026 bracket seeds ──────────────────────────────────────────
def load_seeds():
    """
    Try to load 2026 seeds from Kaggle MNCAATourneySeeds.csv.
    If 2026 is not present, build synthetic seeds from NET rankings.

    Returns: dict mapping seed_str → team_name
             e.g. {"W01": "Duke", "W02": "Auburn", ...}
    """
    # ── Try Kaggle seeds first ────────────────────────────────────────────────
    try:
        seeds_df  = pd.read_csv(f"{KAGGLE_DIR}/MNCAATourneySeeds.csv")
        teams_df  = pd.read_csv(f"{KAGGLE_DIR}/MTeams.csv")
        spell_df  = pd.read_csv(f"{KAGGLE_DIR}/MTeamSpellings.csv")

        s2026 = seeds_df[seeds_df["Season"] == 2026]
        if len(s2026) >= 64:
            id_to_name = {int(r["TeamID"]): str(r["TeamName"])
                          for _, r in teams_df.iterrows()}
            seed_map = {}
            for _, row in s2026.iterrows():
                seed_str = str(row["Seed"])         # e.g. "W01", "W16a"
                seed_key = seed_str[:3]              # strip play-in suffix
                team_id  = int(row["TeamID"])
                team_name = id_to_name.get(team_id, f"Team_{team_id}")
                if seed_key not in seed_map:         # keep first (avoid play-in dups)
                    seed_map[seed_key] = team_name
            print(f"Loaded {len(seed_map)} 2026 seeds from Kaggle file.")
            return seed_map, "kaggle"
    except Exception as e:
        print(f"  [!] Could not load Kaggle 2026 seeds: {e}")

    # ── Fallback: build synthetic seeds from NCAA NET rankings ───────────────
    print("  Falling back to NET-rankings-based seeding.")
    return _build_net_seeds(), "net_fallback"


def _build_net_seeds():
    """
    Assign seeds 1–16 across 4 regions using the top 68 NET-ranked teams.
    Standard S-curve seeding: top 4 are #1 seeds, next 4 are #2 seeds, etc.
    Play-in teams (65-68) assigned to the 16-seed slots.
    """
    try:
        net = pd.read_csv(NET_PATH)
        # Try common column names for school name
        name_col = next((c for c in net.columns
                         if c.lower() in ("school", "team", "name")), net.columns[1])
        teams = net[name_col].astype(str).str.strip().tolist()
    except Exception as e:
        print(f"    [!] Could not load NET: {e}  Using placeholder names.")
        teams = [f"Team_{i}" for i in range(1, 69)]

    teams = teams[:68]
    seed_map = {}
    region_order = REGIONS  # W, X, Y, Z

    # S-curve: seed 1 → regions W X Y Z, seed 2 → Z Y X W, seed 3 → W X Y Z, ...
    # Simplified: just assign sequentially for now
    team_idx = 0
    for seed_num in range(1, 17):
        for region in region_order:
            if team_idx >= len(teams):
                seed_map[f"{region}{seed_num:02d}"] = f"TBD_{team_idx}"
            else:
                seed_map[f"{region}{seed_num:02d}"] = teams[team_idx]
            team_idx += 1

    print(f"  Synthetic seed map built for {len(seed_map)} slots.")
    return seed_map


# ══════════════════════════════════════════════════════════════════════════════
# Bracket simulation engine
# ══════════════════════════════════════════════════════════════════════════════
def parse_seed_num(seed_str):
    """'W07' → 7"""
    return int(seed_str[1:3])


def simulate_region(region_teams, win_prob_fn, rng):
    """
    Simulate a single 16-team single-elimination region.
    region_teams: list of 16 team names in seed order [seed1, seed2, ..., seed16]
                  (index 0 = seed 1, index 15 = seed 16)
    Returns: list of 4 round winners for this region (rounds 1-4 → round 4 = Elite 8 winner)
    """
    # Round of 64: 1v16, 2v15, 3v14, 4v13, 5v12, 6v11, 7v10, 8v9
    bracket = list(region_teams)   # copy
    round_winners = []

    # 4 rounds within a region
    while len(bracket) > 1:
        next_round = []
        for i in range(0, len(bracket), 2):
            a, b = bracket[i], bracket[i + 1]
            p    = win_prob_fn(a, b)
            winner = a if rng.random() < p else b
            next_round.append(winner)
        round_winners.append(next_round)
        bracket = next_round

    return round_winners  # list of [r1_winners(8), r2_winners(4), r3_winners(2), r4_winner(1)]


def run_single_tournament(seed_map, win_prob_fn, rng):
    """
    Simulate one full 64-team tournament.
    Returns: dict mapping round_number → list of winners reaching that round.
    """
    # Build region brackets (seed order: 1,16,8,9,5,12,4,13,6,11,3,14,7,10,2,15)
    # Standard bracket ordering within a region
    REGION_BRACKET_ORDER = [1,16,8,9,5,12,4,13,6,11,3,14,7,10,2,15]

    region_elites = []  # 4 Elite 8 winners (one per region)
    all_rounds    = defaultdict(list)  # round → [teams reaching that round]

    for region in REGIONS:
        bracket = []
        for s in REGION_BRACKET_ORDER:
            key  = f"{region}{s:02d}"
            team = seed_map.get(key, f"TBD_{region}{s:02d}")
            bracket.append(team)

        # All teams start in round 1
        all_rounds[1].extend(bracket)

        round_results = simulate_region(bracket, win_prob_fn, rng)
        # round_results[0] = 8 teams after R1, [1] = 4 after R2, [2] = 2 after R3, [3] = 1 Elite8

        for r_idx, winners in enumerate(round_results):
            round_num = r_idx + 2  # 2=R32, 3=S16, 4=E8
            all_rounds[round_num].extend(winners)

        region_elites.append(round_results[-1][0])

    # Final Four (round 5): W vs X, Y vs Z
    ff_winners = []
    for i in range(0, 4, 2):
        a, b = region_elites[i], region_elites[i + 1]
        p    = win_prob_fn(a, b)
        ff_winners.append(a if rng.random() < p else b)
    all_rounds[5].extend(ff_winners)

    # Championship (round 6)
    a, b = ff_winners[0], ff_winners[1]
    p    = win_prob_fn(a, b)
    champ = a if rng.random() < p else b
    all_rounds[6].append(champ)

    return dict(all_rounds)


def run_simulations(seed_map, win_prob_fn, n_sims=DEFAULT_SIMS, seed=42):
    """Run n_sims full tournaments. Return per-team round reach counts."""
    rng = np.random.default_rng(seed)

    all_teams     = set(seed_map.values())
    reach_counts  = {team: defaultdict(int) for team in all_teams}
    raw_rows      = []  # for saving sample outcomes

    for i in range(n_sims):
        result = run_single_tournament(seed_map, win_prob_fn, rng)
        for rnd, winners in result.items():
            for team in winners:
                reach_counts[team][rnd] += 1

        # Save first 10k simulations for raw output
        if i < 10_000:
            champ = result.get(6, ["?"])[0] if result.get(6) else "?"
            ff    = result.get(5, [])
            raw_rows.append({"sim": i, "champion": champ,
                             "final4": "|".join(ff[:4])})

    # Convert to probabilities
    records = []
    for team in all_teams:
        row = {"team": team}
        for rnd in range(1, 7):
            row[ROUND_NAMES[rnd]] = round(reach_counts[team][rnd] / n_sims, 6)
        records.append(row)

    round_probs_df = pd.DataFrame(records).sort_values(
        ROUND_NAMES[6], ascending=False).reset_index(drop=True)
    raw_df = pd.DataFrame(raw_rows)

    return round_probs_df, raw_df, reach_counts


# ══════════════════════════════════════════════════════════════════════════════
# Upset path correlation analysis
# ══════════════════════════════════════════════════════════════════════════════
def analyze_upset_paths(seed_map, win_prob_fn, reach_counts, n_sims):
    """
    Identify teams with seed ≥ 10 that have a non-trivial championship path.
    Rank by P(reach Final Four).
    """
    upset_records = []
    for key, team in seed_map.items():
        seed_num = parse_seed_num(key)
        if seed_num < 10:
            continue
        region = key[0]
        p_ff   = reach_counts[team][5] / n_sims
        p_e8   = reach_counts[team][4] / n_sims
        p_s16  = reach_counts[team][3] / n_sims
        if p_s16 < 0.01:
            continue
        upset_records.append({
            "team": team, "seed": seed_num, "region": REGION_NAMES.get(region, region),
            "P(Sweet16)": round(p_s16, 4),
            "P(Elite8)":  round(p_e8, 4),
            "P(FinalFour)": round(p_ff, 4),
        })

    upset_df = pd.DataFrame(upset_records).sort_values("P(FinalFour)", ascending=False)
    return upset_df


# ══════════════════════════════════════════════════════════════════════════════
# Pool EV optimization
# ══════════════════════════════════════════════════════════════════════════════
def score_bracket(bracket_picks, tournament_result, scoring, upset_mult=1.0, seed_map=None):
    """
    Score a bracket against one simulated tournament.
    bracket_picks: dict round → list of picked winners
    tournament_result: dict round → list of actual winners
    scoring: list of points per round [r1, r2, r3, r4, r5, r6]
    """
    total = 0.0
    for rnd in range(1, 7):
        base_pts  = scoring[rnd - 1] if rnd <= len(scoring) else scoring[-1]
        picked    = set(bracket_picks.get(rnd, []))
        actual    = set(tournament_result.get(rnd, []))
        hits      = picked & actual

        for team in hits:
            pts = base_pts
            if upset_mult > 1.0 and seed_map:
                # Find this team's seed
                team_seed = next(
                    (parse_seed_num(k) for k, v in seed_map.items() if v == team), 1
                )
                if team_seed >= 10:
                    pts *= upset_mult
            total += pts
    return total


def build_chalk_bracket(seed_map, win_prob_fn):
    """
    Chalk strategy: always pick the team with higher P(win).
    Returns dict round → list of picked winners.
    """
    def best_pick(a, b):
        return a if win_prob_fn(a, b) >= 0.5 else b

    picks = {}
    REGION_BRACKET_ORDER = [1,16,8,9,5,12,4,13,6,11,3,14,7,10,2,15]

    region_elites = []
    for region in REGIONS:
        bracket = [seed_map.get(f"{region}{s:02d}", f"TBD_{region}{s:02d}")
                   for s in REGION_BRACKET_ORDER]

        picks.setdefault(1, []).extend(bracket)
        round_idx = 2
        while len(bracket) > 1:
            next_round = [best_pick(bracket[i], bracket[i+1])
                          for i in range(0, len(bracket), 2)]
            picks.setdefault(round_idx, []).extend(next_round)
            bracket = next_round
            round_idx += 1

        region_elites.append(bracket[0])

    # Final Four
    ff = [best_pick(region_elites[0], region_elites[1]),
          best_pick(region_elites[2], region_elites[3])]
    picks[5] = ff
    picks[6] = [best_pick(ff[0], ff[1])]

    return picks


def build_exploitative_bracket(seed_map, win_prob_fn, round_probs_df,
                                 public_col=None, n_contrarian=4):
    """
    Exploitative strategy: tilt toward under-picked teams.
    In absence of public pick % data, we use teams whose model probability
    significantly exceeds their seed expectation (i.e., model likes them
    more than a pure-seed baseline would).

    We replace up to n_contrarian picks with model-favoured mid-major surprises.
    """
    chalk = build_chalk_bracket(seed_map, win_prob_fn)

    # Identify strong contrarian candidates: seed ≥ 5 but high P(Elite 8)
    contras = round_probs_df[
        round_probs_df["team"].apply(
            lambda t: next((parse_seed_num(k) for k, v in seed_map.items() if v == t), 1)
        ) >= 5
    ].sort_values(ROUND_NAMES[4], ascending=False).head(n_contrarian)["team"].tolist()

    exploit = {r: list(picks) for r, picks in chalk.items()}

    # In rounds 2-4, swap chalk picks for contrarians where model EV > chalk
    for rnd in [2, 3, 4]:
        current = exploit.get(rnd, [])
        for team in contras:
            if team in current:
                continue  # already picked
            # Find who the contrarian team would replace (their region bracket slot)
            region = next((k[0] for k, v in seed_map.items() if v == team), None)
            if region is None:
                continue
            # Replace lowest-probability pick in same region
            regional = [t for t in current
                        if any(v == t and k.startswith(region)
                               for k, v in seed_map.items())]
            if not regional:
                continue
            worst = min(regional, key=lambda t: win_prob_fn(t, team))
            idx   = current.index(worst)
            current[idx] = team
        exploit[rnd] = current

    return exploit


def build_risk_adjusted_brackets(seed_map, win_prob_fn, round_probs_df, n=3):
    """
    Return n brackets with different risk levels:
      - low risk:  chalk-heavy
      - medium:    moderate contrarianism
      - high risk: max variance (maximize expected pool spread)
    """
    chalk = build_chalk_bracket(seed_map, win_prob_fn)
    med   = build_exploitative_bracket(seed_map, win_prob_fn, round_probs_df, n_contrarian=3)
    high  = build_exploitative_bracket(seed_map, win_prob_fn, round_probs_df, n_contrarian=8)
    return {"low_risk_chalk": chalk, "medium_risk": med, "high_risk": high}


def estimate_pool_ev(brackets, seed_map, win_prob_fn, n_sims=5_000,
                     scoring=DEFAULT_SCORING, upset_mult=DEFAULT_UPSET_MULT):
    """
    Monte Carlo: simulate n_sims tournaments, score each bracket.
    Returns DataFrame with mean/std/percentile expected points per bracket.
    """
    rng     = np.random.default_rng(seed=99)
    results = {name: [] for name in brackets}

    for _ in range(n_sims):
        sim = run_single_tournament(seed_map, win_prob_fn, rng)
        for name, picks in brackets.items():
            s = score_bracket(picks, sim, scoring, upset_mult, seed_map)
            results[name].append(s)

    records = []
    for name, scores in results.items():
        scores = np.array(scores)
        records.append({
            "bracket":         name,
            "mean_ev":         round(float(np.mean(scores)), 2),
            "std_ev":          round(float(np.std(scores)), 2),
            "p10":             round(float(np.percentile(scores, 10)), 1),
            "p50":             round(float(np.percentile(scores, 50)), 1),
            "p90":             round(float(np.percentile(scores, 90)), 1),
            "p99":             round(float(np.percentile(scores, 99)), 1),
        })

    ev_df = pd.DataFrame(records).sort_values("mean_ev", ascending=False)
    return ev_df


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_results(round_probs_df, ev_df, upset_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── Panel 1: Top 20 teams by championship probability ──────────────────
    ax = axes[0]
    top20 = round_probs_df.head(20)
    ax.barh(top20["team"][::-1], top20[ROUND_NAMES[6]][::-1], color="royalblue")
    ax.set_xlabel("P(Win Championship)")
    ax.set_title("Top 20: Championship Win Probability")
    ax.grid(True, alpha=0.3, axis="x")

    # ── Panel 2: Round reach for top 8 ────────────────────────────────────
    ax = axes[1]
    top8     = round_probs_df.head(8)
    rounds   = [ROUND_NAMES[r] for r in range(2, 7)]
    x        = np.arange(len(rounds))
    width    = 0.8 / len(top8)
    for i, (_, row) in enumerate(top8.iterrows()):
        vals = [row[r] for r in rounds]
        ax.bar(x + i * width, vals, width, label=row["team"], alpha=0.8)
    ax.set_xticks(x + width * len(top8) / 2)
    ax.set_xticklabels([r.replace(" ", "\n") for r in rounds], fontsize=7)
    ax.set_ylabel("Probability")
    ax.set_title("Round Reach — Top 8 Teams")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 3: Pool EV comparison across bracket strategies ─────────────
    ax = axes[2]
    ax.barh(ev_df["bracket"], ev_df["mean_ev"], xerr=ev_df["std_ev"],
            color=["steelblue","darkorange","green"], capsize=4)
    ax.set_xlabel("Expected Pool Points (mean ± std)")
    ax.set_title("Pool EV by Bracket Strategy")
    ax.grid(True, alpha=0.3, axis="x")
    for _, row in ev_df.iterrows():
        ax.text(row["mean_ev"] + 1, ev_df["bracket"].tolist().index(row["bracket"]),
                f'p90={row["p90"]:.0f}', va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(SIM_PLOTS_OUT, dpi=120)
    plt.close()
    print(f"Simulation plots saved → {SIM_PLOTS_OUT}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims",             type=int,   default=DEFAULT_SIMS)
    parser.add_argument("--scoring",          type=str,   default="1,2,4,8,16,32")
    parser.add_argument("--upset-multiplier", type=float, default=DEFAULT_UPSET_MULT)
    args = parser.parse_args()

    scoring    = [int(x) for x in args.scoring.split(",")]
    upset_mult = args.upset_multiplier
    n_sims     = args.sims

    print("=" * 60)
    print(f"Phase 6 — Tournament Simulation  (n={n_sims:,})")
    print("=" * 60)

    # ── Load ensemble probs ───────────────────────────────────────────────────
    try:
        ensemble_df = pd.read_csv(ENSEMBLE_PROBS)
    except FileNotFoundError:
        print(f"[!] {ENSEMBLE_PROBS} not found — run phase5 first.")
        return

    win_prob_fn = build_prob_lookup(ensemble_df)

    # ── Load bracket seeds ────────────────────────────────────────────────────
    seed_map, seed_source = load_seeds()
    print(f"  Seed source: {seed_source}  ({len(seed_map)} slots)")

    n_known = sum(1 for t in seed_map.values() if not t.startswith("TBD"))
    print(f"  Teams with known names: {n_known} / {len(seed_map)}")
    if n_known < 50:
        print("  [!] Many teams are TBD — win probs will default to 0.5 for unknown matchups.")

    # ── Full tournament simulation ────────────────────────────────────────────
    print(f"\nSimulating {n_sims:,} tournaments ...")
    round_probs_df, raw_df, reach_counts = run_simulations(seed_map, win_prob_fn, n_sims)

    round_probs_df.to_csv(ROUND_PROBS_OUT, index=False)
    raw_df.to_csv(SIM_RAW_OUT, index=False)
    print(f"Team round probabilities → {ROUND_PROBS_OUT}")
    print(f"Raw simulation sample    → {SIM_RAW_OUT}")

    # ── Top contenders ────────────────────────────────────────────────────────
    print(f"\n── Top 10 Championship Contenders ──")
    print(round_probs_df[["team", ROUND_NAMES[4], ROUND_NAMES[5],
                           ROUND_NAMES[6]]].head(10).to_string(index=False))

    # ── Upset path analysis ────────────────────────────────────────────────────
    print(f"\n── Key Upset Paths ──")
    upset_df = analyze_upset_paths(seed_map, win_prob_fn, reach_counts, n_sims)
    upset_df.to_csv(UPSET_PATHS_OUT, index=False)
    print(upset_df.head(10).to_string(index=False))
    print(f"Upset paths saved → {UPSET_PATHS_OUT}")

    # ── Bracket strategies ─────────────────────────────────────────────────────
    print(f"\n── Building bracket strategies (scoring={scoring}, upset_mult={upset_mult}) ──")
    brackets = build_risk_adjusted_brackets(seed_map, win_prob_fn, round_probs_df)

    # Add explicit chalk and exploitative labels
    brackets["chalk"]        = brackets.pop("low_risk_chalk")
    brackets["exploitative"] = brackets.pop("medium_risk")
    brackets["high_variance"]= brackets.pop("high_risk")

    with open(BRACKETS_OUT, "w") as f:
        json.dump(brackets, f, indent=2)
    print(f"Bracket picks saved → {BRACKETS_OUT}")

    for name, picks in brackets.items():
        champ = picks.get(6, ["?"])[0] if picks.get(6) else "?"
        ff    = picks.get(5, [])
        print(f"  {name:20s}  champion={champ:<25s}  final4={', '.join(ff)}")

    # ── Pool EV estimation ─────────────────────────────────────────────────────
    print(f"\n── Pool EV estimation (5,000 scoring simulations) ──")
    ev_df = estimate_pool_ev(brackets, seed_map, win_prob_fn,
                             n_sims=5_000, scoring=scoring, upset_mult=upset_mult)
    ev_df.to_csv(POOL_EV_OUT, index=False)
    print(ev_df[["bracket","mean_ev","std_ev","p10","p50","p90","p99"]].to_string(index=False))
    print(f"Pool EV saved → {POOL_EV_OUT}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_results(round_probs_df, ev_df, upset_df)

    print("\n✓ Phase 6 complete.")
    print(f"  Review bracket strategies in {BRACKETS_OUT}")
    print(f"  Pool EV comparison in {POOL_EV_OUT}")
    print(f"  Adjust --scoring and --upset-multiplier for your pool's rules.")


if __name__ == "__main__":
    main()

"""
utils/live_scorer.py
Ingests actual 2026 tournament results and computes live log-loss tracking.
Returns a safe zero-structure when the tournament hasn't started yet.
"""
import numpy as np
import pandas as pd
from pathlib import Path

_ROOT    = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _ROOT / "data"

# Empirical historical seed win rates (higher seed = better team)
SEED_IMPLIED_RATES: dict[tuple, float] = {
    (1, 16): 0.99, (2, 15): 0.94, (3, 14): 0.85, (4, 13): 0.79,
    (5, 12): 0.65, (6, 11): 0.62, (7, 10): 0.61, (8,  9): 0.49,
}

ROUND_MAP = {1: "Round of 64", 2: "Round of 32", 3: "Sweet 16",
             4: "Elite 8",     5: "Final Four",  6: "Championship"}

EMPTY_RESULT = {
    "tournament_started": False,
    "games_played":       0,
    "model_log_loss":     None,
    "baseline_log_loss":  None,
    "model_brier":        None,
    "model_accuracy":     None,
    "per_round_log_loss": {},
    "correct_picks":      0,
    "total_picks":        0,
    "log_loss_by_game":   [],
    "model_beats_baseline": False,
}


def _safe_ll(y_true: float, p_pred: float) -> float:
    p = max(1e-7, min(1 - 1e-7, p_pred))
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def compute_live_metrics(
    df_actual: pd.DataFrame,
    df_pairwise: pd.DataFrame,
) -> dict:
    """
    For each completed 2026 game in df_actual, look up model's win probability
    and compute running log-loss / Brier / accuracy vs a seed-only baseline.

    df_actual columns (from MNCAATourneyDetailedResults): Season, DayNum,
        WTeamID, WScore, LTeamID, LScore, NumOT, DayNum → round number.

    df_pairwise columns: team_a, team_b, ensemble_prob, seed_num_a, seed_num_b
    """
    if df_actual is None or len(df_actual) == 0:
        return EMPTY_RESULT

    # Build name→seed and name→id lookups from pairwise df
    prob_lookup: dict[tuple, float] = {}
    for _, r in df_pairwise.iterrows():
        prob_lookup[(r["team_a"], r["team_b"])] = float(r["ensemble_prob"])
        prob_lookup[(r["team_b"], r["team_a"])] = 1 - float(r["ensemble_prob"])

    # Build TeamID→team_name from MTeams
    teams_path = DATA_DIR / "march-machine-learning-mania-2026" / "MTeams.csv"
    seeds_path = DATA_DIR / "march-machine-learning-mania-2026" / "MNCAATourneySeeds.csv"
    id_to_name: dict[int, str] = {}
    id_to_seed: dict[int, int] = {}
    if teams_path.exists():
        mteams = pd.read_csv(teams_path)
        id_to_name = mteams.set_index("TeamID")["TeamName"].to_dict()
    if seeds_path.exists():
        mseeds = pd.read_csv(seeds_path)
        mseeds26 = mseeds[mseeds["Season"] == 2026] if "Season" in mseeds.columns else pd.DataFrame()
        for _, s in mseeds26.iterrows():
            try:
                seed_num = int(str(s["Seed"])[1:3])
                id_to_seed[int(s["TeamID"])] = seed_num
            except Exception:
                pass

    # Assign rounds (rough DayNum → round mapping)
    def daynum_to_round(d: int) -> int:
        if d <= 136: return 1
        if d <= 138: return 2
        if d <= 143: return 3
        if d <= 145: return 4
        if d <= 152: return 5
        return 6

    games, model_lls, base_lls, brierr, correct = [], [], [], [], []
    per_round: dict[int, list] = {r: [] for r in range(1, 7)}

    for _, game in df_actual.iterrows():
        w_id = int(game["WTeamID"])
        l_id = int(game["LTeamID"])
        w_name = id_to_name.get(w_id, str(w_id))
        l_name = id_to_name.get(l_id, str(l_id))
        rnd    = daynum_to_round(int(game.get("DayNum", 130)))

        # Model probability that winner beats loser
        p_model = prob_lookup.get((w_name, l_name),
                  1 - prob_lookup.get((l_name, w_name), 0.5))

        # Seed-implied baseline
        sw = id_to_seed.get(w_id, 8)
        sl = id_to_seed.get(l_id, 8)
        key = (min(sw, sl), max(sw, sl))
        seed_rate = SEED_IMPLIED_RATES.get(key, 0.5)
        p_base = seed_rate if sw < sl else 1 - seed_rate

        y = 1.0  # winner always wins
        ll_m = _safe_ll(y, p_model)
        ll_b = _safe_ll(y, p_base)
        bs   = (y - p_model) ** 2

        model_lls.append(ll_m)
        base_lls.append(ll_b)
        brierr.append(bs)
        correct.append(1 if p_model >= 0.5 else 0)
        per_round[rnd].append(ll_m)
        games.append({"game": f"{w_name} def. {l_name}", "round": ROUND_MAP.get(rnd, str(rnd)),
                      "p_model": round(p_model, 3), "log_loss": round(ll_m, 4)})

    if not model_lls:
        return EMPTY_RESULT

    avg_ll_m = float(np.mean(model_lls))
    avg_ll_b = float(np.mean(base_lls))

    return {
        "tournament_started":   True,
        "games_played":         len(model_lls),
        "model_log_loss":       round(avg_ll_m, 4),
        "baseline_log_loss":    round(avg_ll_b, 4),
        "model_brier":          round(float(np.mean(brierr)), 4),
        "model_accuracy":       round(float(np.mean(correct)), 4),
        "per_round_log_loss":   {ROUND_MAP.get(r, str(r)): round(float(np.mean(v)), 4)
                                 for r, v in per_round.items() if v},
        "correct_picks":        sum(correct),
        "total_picks":          len(correct),
        "log_loss_by_game":     games,
        "model_beats_baseline": avg_ll_m < avg_ll_b,
    }

"""
utils/sim_engine.py
Vectorized Monte Carlo bracket simulation.
Target: 10,000 sims in < 3 seconds via numpy advanced indexing.
"""
import numpy as np
import pandas as pd
import streamlit as st

# ── Bracket structure ──────────────────────────────────────────────────────────
# Standard NCAA bracket ordering within a region (16 positions, top → bottom)
BRACKET_SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
REGIONS = ["W", "X", "Y", "Z"]

# Game pairs within a region for each round (0-indexed slot positions within region)
# Each entry: (slot_a, slot_b) → winner goes to parent slot
R64_PAIRS = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15)]

# R32: pairs of R64 winner slots (by R64 game index)
# R2W1 = G0 vs G7, R2W2 = G2 vs G5, R2W3 = G4 vs G3, R2W4 = G6 vs G1
# Actually from MNCAATourneySlots: R2W1=R1W1 vs R1W8, R2W2=R1W2 vs R1W7, etc.
R32_PAIRS = [(0, 7), (1, 6), (2, 5), (3, 4)]  # indices into R64_PAIRS winners

# S16: pairs of R32 winner indices
S16_PAIRS  = [(0, 3), (1, 2)]  # R3W1=R2W1 vs R2W4, R3W2=R2W2 vs R2W3

# E8: single pair
E8_PAIRS   = [(0, 1)]

# Final Four: W vs X, Y vs Z
FF_PAIRS   = [(0, 1), (2, 3)]  # region indices

# Championship: 1 game
CHAMP_PAIRS = [(0, 1)]


def _build_prob_matrix(probs_df: pd.DataFrame, team_list: list) -> np.ndarray:
    """
    Build an n×n probability matrix where P[i,j] = P(team i beats team j).
    Teams not found in probs_df default to 0.5.
    """
    n = len(team_list)
    idx = {t: i for i, t in enumerate(team_list)}
    P = np.full((n, n), 0.5)
    np.fill_diagonal(P, 0.5)

    for _, row in probs_df.iterrows():
        a, b = row["team_a"], row["team_b"]
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            p = float(row["ensemble_prob"])
            P[i, j] = p
            P[j, i] = 1 - p
    return P


def _get_region_seeds(region: str, teams_df: pd.DataFrame) -> list:
    """
    Return ordered list of team names for a region, in bracket order
    (slot 0=seed1, slot 1=seed16, slot 2=seed8, … per BRACKET_SEED_ORDER).
    """
    region_teams = teams_df[teams_df["region"] == region].copy()
    seed_to_name = region_teams.set_index("seed_num")["team_name"].to_dict()
    ordered = []
    for seed in BRACKET_SEED_ORDER:
        name = seed_to_name.get(seed)
        if name is None:
            # Fill with a placeholder if seed not assigned
            name = region_teams.iloc[len(ordered) % max(len(region_teams), 1)]["team_name"] \
                   if len(region_teams) > 0 else f"{region}{seed:02d}_TBD"
        ordered.append(name)
    return ordered[:16]


@st.cache_data(ttl=3600, show_spinner=False)
def run_simulation(
    probs_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    n_sims: int = 10_000,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Run n_sims full 64-team bracket simulations using vectorized numpy.

    Returns DataFrame with columns:
      team_name, seed_num, conf, region_name,
      rd1_pct, rd2_pct, rd3_pct, rd4_pct, rd5_pct, rd6_pct (champion)
    """
    rng = np.random.default_rng(rng_seed)

    # Build ordered team list and probability matrix
    all_teams = teams_df.head(64)["team_name"].tolist()
    P = _build_prob_matrix(probs_df, all_teams)
    team_idx = {t: i for i, t in enumerate(all_teams)}

    # For each region, get the 16-team slot order
    region_slots: dict[str, list] = {}
    for reg in REGIONS:
        region_slots[reg] = _get_region_seeds(reg, teams_df.head(64))

    # Flatten all 64 initial team indices into a (64,) array
    # Layout: W[0..15], X[16..31], Y[32..47], Z[48..63]
    initial_teams = np.array([
        team_idx.get(t, 0)
        for reg in REGIONS
        for t in region_slots[reg]
    ], dtype=np.int32)  # shape (64,)

    # sim_slots[sim_idx, slot_idx] = team index currently in that slot
    # We'll process round-by-round, reducing the number of active slots each round
    # For round tracking, keep a (n_sims, 64) array of team indices
    sim_slots = np.tile(initial_teams, (n_sims, 1))  # (n_sims, 64)

    # Track reach counts: reach[team_idx, round] counts how many sims that team reached that round
    reach = np.zeros((64, 7), dtype=np.int32)  # rounds 1-6 (index 0 unused)

    # Round 1 (R64): all 64 teams start here
    for i in range(64):
        reach[:, 0] = 0  # we'll count differently below

    # Helper: simulate one round across all sims
    # games_list: list of (slot_a, slot_b) into sim_slots
    # Returns: updated sim_slots (same shape, winners placed in slot_a positions)
    def sim_round(sim_slots_in: np.ndarray, games: list) -> tuple:
        """Returns (winners_by_game: shape (n_sims, n_games), sim_slots_out)"""
        n_games = len(games)
        winners_out = np.zeros((n_sims, n_games), dtype=np.int32)
        sim_slots_out = sim_slots_in.copy()

        for g_idx, (sa, sb) in enumerate(games):
            teams_a = sim_slots_in[:, sa]  # (n_sims,)
            teams_b = sim_slots_in[:, sb]  # (n_sims,)
            probs_a = P[teams_a, teams_b]   # (n_sims,) — advanced indexing
            draws   = rng.random(n_sims)
            a_wins  = draws < probs_a
            winners = np.where(a_wins, teams_a, teams_b)
            winners_out[:, g_idx] = winners
            sim_slots_out[:, sa]  = winners  # winner stays in slot_a

        return winners_out, sim_slots_out

    # ── R64 (round 1): slots 0-15, 16-31, 32-47, 48-63 ──────────────────────
    r64_games = []
    for reg_offset in [0, 16, 32, 48]:
        for sa, sb in R64_PAIRS:
            r64_games.append((reg_offset + sa, reg_offset + sb))

    r64_winners, sim_slots = sim_round(sim_slots, r64_games)
    # All teams at R64 (we count them all as reaching R1)
    for t in range(64):
        reach[t, 1] = n_sims  # all 64 teams start

    # R64 winners (32 teams reached R32)
    for g in range(32):
        vals, cnts = np.unique(r64_winners[:, g], return_counts=True)
        for v, c in zip(vals, cnts):
            reach[v, 2] += c

    # ── R32 (round 2): use R32_PAIRS across the winners from R64 ─────────────
    r32_games = []
    for reg_offset in [0, 16, 32, 48]:
        # After R64, the winners are in slots [reg_offset+0, reg_offset+2, reg_offset+4, ...]
        # (every even slot within the region's 16 slots held the "a" slot)
        # In sim_slots, after sim_round, slot reg_offset+sa holds winner
        # The R32 pairs match up by game index
        for idx_a, idx_b in R32_PAIRS:
            sa_r64 = R64_PAIRS[idx_a][0]  # slot that holds winner of r64 game idx_a
            sb_r64 = R64_PAIRS[idx_b][0]
            r32_games.append((reg_offset + sa_r64, reg_offset + sb_r64))

    r32_winners, sim_slots = sim_round(sim_slots, r32_games)
    for g in range(16):
        vals, cnts = np.unique(r32_winners[:, g], return_counts=True)
        for v, c in zip(vals, cnts):
            reach[v, 3] += c  # Sweet 16

    # ── S16 (round 3) ────────────────────────────────────────────────────────
    s16_games = []
    for reg_offset in [0, 16, 32, 48]:
        # S16 pairs: winners of R32 games 0&3, 1&2 within region
        r32_slots_in_region = [R64_PAIRS[R32_PAIRS[i][0]][0] for i in range(4)]
        for idx_a, idx_b in S16_PAIRS:
            sa = r32_slots_in_region[idx_a]
            sb = r32_slots_in_region[idx_b]
            s16_games.append((reg_offset + sa, reg_offset + sb))

    s16_winners, sim_slots = sim_round(sim_slots, s16_games)
    for g in range(8):
        vals, cnts = np.unique(s16_winners[:, g], return_counts=True)
        for v, c in zip(vals, cnts):
            reach[v, 4] += c  # Elite 8

    # ── E8 (round 4) ─────────────────────────────────────────────────────────
    e8_games = []
    for reg_offset in [0, 16, 32, 48]:
        s16_slots = [R64_PAIRS[R32_PAIRS[S16_PAIRS[i][0]][0]][0] for i in range(2)]
        e8_games.append((reg_offset + s16_slots[0], reg_offset + s16_slots[1]))

    e8_winners, sim_slots = sim_round(sim_slots, e8_games)  # 4 winners (one per region)
    for g in range(4):
        vals, cnts = np.unique(e8_winners[:, g], return_counts=True)
        for v, c in zip(vals, cnts):
            reach[v, 5] += c  # Final Four

    # ── Final Four (round 5) ─────────────────────────────────────────────────
    # E8 winners stored in e8_winners[:, 0..3] for regions W,X,Y,Z
    # FF: W vs X (games 0,1), Y vs Z (games 2,3)
    ff_a = e8_winners[:, 0]  # W winner
    ff_b = e8_winners[:, 1]  # X winner
    ff_c = e8_winners[:, 2]  # Y winner
    ff_d = e8_winners[:, 3]  # Z winner

    p_ff1 = P[ff_a, ff_b]
    p_ff2 = P[ff_c, ff_d]
    ff1_wins = rng.random(n_sims) < p_ff1
    ff2_wins = rng.random(n_sims) < p_ff2
    champ_left  = np.where(ff1_wins, ff_a, ff_b)
    champ_right = np.where(ff2_wins, ff_c, ff_d)

    for t in np.unique(champ_left):
        reach[t, 6] += int(np.sum(champ_left == t))  # but we need to be careful — this is the champ game not final outcome
    # Redo: count all FF participants
    for arr in [ff_a, ff_b, ff_c, ff_d]:
        vals, cnts = np.unique(arr, return_counts=True)
        for v, c in zip(vals, cnts):
            pass  # already counted above

    # ── Championship (round 6) ────────────────────────────────────────────────
    p_champ = P[champ_left, champ_right]
    champ_wins = rng.random(n_sims) < p_champ
    champions  = np.where(champ_wins, champ_left, champ_right)

    vals, cnts = np.unique(champions, return_counts=True)
    for v, c in zip(vals, cnts):
        reach[v, 6] = c  # overwrite with championship count

    # ── Build output DataFrame ────────────────────────────────────────────────
    records = []
    teams_info = teams_df.head(64).reset_index(drop=True)
    for t_idx, team_name in enumerate(all_teams):
        row_info = teams_info[teams_info["team_name"] == team_name]
        seed_num   = int(row_info["seed_num"].values[0])   if len(row_info) else 0
        conf       = str(row_info["conf"].values[0])       if len(row_info) else ""
        region_nm  = str(row_info["region_name"].values[0]) if len(row_info) else ""

        records.append({
            "team_name":   team_name,
            "seed_num":    seed_num,
            "conf":        conf,
            "region_name": region_nm,
            "rd1_pct":  100.0,                           # everyone reaches R64
            "rd2_pct":  100 * reach[t_idx, 2] / n_sims, # R32
            "rd3_pct":  100 * reach[t_idx, 3] / n_sims, # S16
            "rd4_pct":  100 * reach[t_idx, 4] / n_sims, # E8
            "rd5_pct":  100 * reach[t_idx, 5] / n_sims, # FF
            "rd6_pct":  100 * reach[t_idx, 6] / n_sims, # Champion
        })

    df = pd.DataFrame(records).sort_values("rd6_pct", ascending=False).reset_index(drop=True)
    return df

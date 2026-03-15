"""
utils/data_loader.py
All @st.cache_data loading functions.
Data is read from actual CSVs on disk; no parquet/pkl files required.

Priority chain for probabilities:
  1. phase5_ensemble_probs.csv  (BMA + stack + risk-adaptive blend — most accurate)
  2. xgb_predictions_2026.csv + baseline_predictions_2026.csv  (fallback)

Priority chain for simulation results:
  1. phase6_team_round_probs.csv  (10k-sim output — already computed)
  2. Run sim_engine in-memory with 5,000 sims  (fallback)
"""
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_APP_DIR   = Path(__file__).resolve().parent.parent   # .../app/
_ROOT      = _APP_DIR.parent                           # project root
DATA_DIR   = _ROOT / "data"
KAGGLE_DIR = DATA_DIR / "march-machine-learning-mania-2026"

# ── Bracket region / seed assignment constants ────────────────────────────────
REGIONS      = ["W", "X", "Y", "Z"]
REGION_NAMES = {"W": "East", "X": "West", "Y": "South", "Z": "Midwest"}

def _seed_region_order(seed_num: int):
    return ["W", "X", "Y", "Z"] if seed_num % 2 == 1 else ["Z", "Y", "X", "W"]

BRACKET_SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]


# ── Private helpers ────────────────────────────────────────────────────────────
def _parse_record(record: str):
    """'24-8' → (24, 8)"""
    try:
        parts = str(record).split("-")
        return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 0


def _assign_seeds(net_df: pd.DataFrame) -> pd.DataFrame:
    df = net_df.head(68).copy().reset_index(drop=True)
    seeds, regions, seed_strs = [], [], []

    for idx, row in df.iterrows():
        slot = idx
        if slot < 64:
            seed_num    = (slot // 4) + 1
            pos_in_seed = slot % 4
            region      = _seed_region_order(seed_num)[pos_in_seed]
        else:
            seed_num = 16
            region   = ["W", "X"][slot - 64] if (slot - 64) < 2 else "Y"

        seeds.append(seed_num)
        regions.append(region)
        seed_strs.append(f"{region}{seed_num:02d}")

    df["seed_num"]    = seeds
    df["region"]      = regions
    df["seed_str"]    = seed_strs
    df["region_name"] = df["region"].map(REGION_NAMES)
    return df


def _load_torvik_2026() -> pd.DataFrame:
    # Prefer the dedicated 2026 advanced file if it exists
    adv_path = DATA_DIR / "barttorvik_adv_2026.csv"
    hist_path = DATA_DIR / "barttorvik_historical.csv"

    for path, is_adv in [(adv_path, True), (hist_path, False)]:
        if not path.exists():
            continue
        try:
            torvik = pd.read_csv(path, low_memory=False)
            rename_map = {
                "0": "rank_t", "1": "team", "2": "conf_t", "3": "record_t",
                "4": "adjoe", "5": "oe_rank", "6": "adjde", "7": "de_rank",
                "8": "barthag", "15": "sos", "16": "ncsos", "17": "consos",
                "41": "wab", "44": "adjt",
            }
            torvik = torvik.rename(columns={k: v for k, v in rename_map.items()
                                            if k in torvik.columns})
            if "team" in torvik.columns:
                torvik = torvik[torvik["team"] != "team"]

            if not is_adv:
                torvik["season"] = pd.to_numeric(
                    torvik.get("season", np.nan), errors="coerce")
                torvik = torvik[torvik["season"] == 2026].copy()
            else:
                torvik = torvik.copy()

            for col in ["adjoe", "adjde", "barthag", "sos", "wab", "adjt"]:
                if col in torvik.columns:
                    torvik[col] = pd.to_numeric(torvik[col], errors="coerce")
            torvik = torvik.drop_duplicates(subset=["team"])
            if len(torvik):
                return torvik.reset_index(drop=True)
        except Exception:
            continue
    return pd.DataFrame()


# ── Public loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_teams() -> pd.DataFrame:
    """
    Returns one row per tournament team (68 teams).
    Columns: team_name, net_rank, conf, record, wins, losses,
             quad1..4, seed_num, region, region_name, seed_str,
             adjoe, adjde, barthag, sos, wab, adjt  (Barttorvik 2026)
             *_pct  (percentile ranks within the 68-team field)
    """
    net = pd.read_csv(DATA_DIR / "ncaa_net.csv")
    net = net.rename(columns={
        "School": "team_name", "Conf": "conf",
        "Record": "record",    "Rank": "net_rank",
        "Quad 1": "quad1",     "Quad 2": "quad2",
        "Quad 3": "quad3",     "Quad 4": "quad4",
    })
    df = _assign_seeds(net)
    df["wins"]   = df["record"].apply(lambda r: _parse_record(r)[0])
    df["losses"] = df["record"].apply(lambda r: _parse_record(r)[1])

    torvik = _load_torvik_2026()
    if len(torvik) and "team" in torvik.columns:
        torvik_sub = torvik[["team", "adjoe", "adjde", "barthag",
                              "sos", "wab", "adjt"]].copy()
        torvik_sub["team_key"] = torvik_sub["team"].str.lower().str.strip()
        df["team_key"] = df["team_name"].str.lower().str.strip()
        df = df.merge(torvik_sub.drop(columns=["team"]), on="team_key", how="left")
        df.drop(columns=["team_key"], inplace=True)
    else:
        for col in ["adjoe", "adjde", "barthag", "sos", "wab", "adjt"]:
            df[col] = np.nan

    # Percentile ranks (0–100, higher = better)
    for col in ["adjoe", "barthag", "sos", "wab", "adjt"]:
        if col in df.columns:
            mn, mx = df[col].min(), df[col].max()
            df[f"{col}_pct"] = 100 * (df[col] - mn) / (mx - mn + 1e-9)
    if "adjde" in df.columns:          # lower DE is better → invert
        mn, mx = df["adjde"].min(), df["adjde"].max()
        df["adjde_pct"] = 100 * (mx - df["adjde"]) / (mx - mn + 1e-9)

    return df.head(68).reset_index(drop=True)


@st.cache_data(ttl=3600)
def load_pairwise_probs() -> pd.DataFrame:
    """
    Load win probabilities for all C(68,2) = 2,016 matchups.

    Priority:
      1. phase5_ensemble_probs.csv  — BMA + stack + risk-adaptive blend
      2. xgb_predictions_2026.csv + baseline_predictions_2026.csv  — simple blend

    Returns DataFrame with columns:
      team_a, team_b, ensemble_prob,
      [xgb_prob, baseline_prob, p_bma, p_stack, p_risk_adaptive]
      prob_lower, prob_upper
      seed_num_a/b, region_a/b, conf_a/b, net_rank_a/b
    """
    p5_path = DATA_DIR / "phase5_ensemble_probs.csv"

    if p5_path.exists():
        # ── Phase 5 BMA ensemble (preferred) ─────────────────────────────────
        df = pd.read_csv(p5_path)
        # Rename to canonical "ensemble_prob" if needed
        if "ensemble_prob" not in df.columns and "p_bma" in df.columns:
            df["ensemble_prob"] = df["ensemble_prob"] if "ensemble_prob" in df.columns \
                                  else df["p_bma"]
        if "ensemble_prob" not in df.columns:
            df["ensemble_prob"] = 0.5

        # Confidence band from disagreement between component models
        comp_cols = [c for c in ["p_bma", "p_stack", "p_risk_adaptive"]
                     if c in df.columns]
        if len(comp_cols) >= 2:
            spread = df[comp_cols].std(axis=1)
        else:
            spread = pd.Series(0.03, index=df.index)

        df["prob_lower"] = (df["ensemble_prob"] - 2 * spread).clip(0.01, 0.99)
        df["prob_upper"] = (df["ensemble_prob"] + 2 * spread).clip(0.01, 0.99)
        df["prob_lower"], df["prob_upper"] = (
            df[["prob_lower", "prob_upper"]].min(axis=1),
            df[["prob_upper", "prob_lower"]].max(axis=1),
        )

    else:
        # ── Fallback: XGB + baseline simple blend ─────────────────────────────
        xgb_path  = DATA_DIR / "xgb_predictions_2026.csv"
        base_path = DATA_DIR / "baseline_predictions_2026.csv"

        if not xgb_path.exists():
            return pd.DataFrame(columns=["team_a", "team_b", "ensemble_prob"])

        xgb_df  = pd.read_csv(xgb_path)
        base_df = pd.read_csv(base_path) if base_path.exists() else None

        if base_df is not None and "team_a" in base_df.columns:
            df = xgb_df.merge(base_df, on=["team_a", "team_b"], how="inner")
            df = df.rename(columns={
                "xgb_prob_a_wins":      "xgb_prob",
                "baseline_prob_a_wins": "baseline_prob",
            })
            df["ensemble_prob"] = 0.6 * df["xgb_prob"] + 0.4 * df["baseline_prob"]
            spread = (df["xgb_prob"] - df["baseline_prob"]).abs()
        else:
            df = xgb_df.copy()
            prob_col = [c for c in df.columns if "prob" in c.lower()]
            df["ensemble_prob"] = df[prob_col[0]] if prob_col else 0.5
            spread = pd.Series(0.03, index=df.index)

        df["prob_lower"] = (df["ensemble_prob"] - 1.5 * spread).clip(0.01, 0.99)
        df["prob_upper"] = (df["ensemble_prob"] + 1.5 * spread).clip(0.01, 0.99)
        df["prob_lower"], df["prob_upper"] = (
            df[["prob_lower", "prob_upper"]].min(axis=1),
            df[["prob_lower", "prob_upper"]].max(axis=1),
        )

    # ── Join team metadata ────────────────────────────────────────────────────
    teams   = load_teams()[["team_name", "seed_num", "region", "conf", "net_rank"]].copy()
    teams_a = teams.rename(columns=lambda c: c + "_a" if c != "team_name" else "team_a")
    teams_b = teams.rename(columns=lambda c: c + "_b" if c != "team_name" else "team_b")
    df = df.merge(teams_a, on="team_a", how="left")
    df = df.merge(teams_b, on="team_b", how="left")
    return df.reset_index(drop=True)


@st.cache_data(ttl=3600)
def load_sim_results() -> pd.DataFrame:
    """
    Return round-reach probabilities for all tournament teams.

    Priority:
      1. phase6_team_round_probs.csv  — pre-computed 10k-sim Phase 6 output
      2. Run sim_engine in-memory with 5,000 sims  (fallback if CSV missing)

    Always returns columns:
      team_name, seed_num, conf, region_name,
      rd1_pct, rd2_pct, rd3_pct, rd4_pct, rd5_pct, rd6_pct
    """
    p6_path = DATA_DIR / "phase6_team_round_probs.csv"

    if p6_path.exists():
        df = pd.read_csv(p6_path)

        # Normalise column names to rd1_pct…rd6_pct schema
        round_map = {
            "Round of 64":  "rd1_pct",
            "Round of 32":  "rd2_pct",
            "Sweet 16":     "rd3_pct",
            "Elite 8":      "rd4_pct",
            "Final Four":   "rd5_pct",
            "Championship": "rd6_pct",
        }
        df = df.rename(columns={k: v for k, v in round_map.items() if k in df.columns})

        # Scale 0–1 → percentage if stored as fractions
        for col in ["rd1_pct","rd2_pct","rd3_pct","rd4_pct","rd5_pct","rd6_pct"]:
            if col in df.columns and df[col].max() <= 1.01:
                df[col] = df[col] * 100

        # Rename "team" → "team_name" if needed
        if "team" in df.columns and "team_name" not in df.columns:
            df = df.rename(columns={"team": "team_name"})

        # Add metadata columns from load_teams if missing
        missing_meta = [c for c in ["seed_num", "conf", "region_name"]
                        if c not in df.columns]
        if missing_meta:
            try:
                teams = load_teams()[["team_name"] + missing_meta]
                df = df.merge(teams, on="team_name", how="left")
            except Exception:
                for c in missing_meta:
                    df[c] = np.nan

        # Sort by champion probability descending
        if "rd6_pct" in df.columns:
            df = df.sort_values("rd6_pct", ascending=False).reset_index(drop=True)
        return df

    # ── Fallback: run sim engine ──────────────────────────────────────────────
    from utils.sim_engine import run_simulation
    teams = load_teams()
    probs = load_pairwise_probs()
    return run_simulation(probs, teams, n_sims=5_000)


@st.cache_data(ttl=3600)
def load_upset_candidates() -> pd.DataFrame:
    """
    Identify upset candidates from Phase 6 upset paths (preferred) or
    by comparing model probs vs seed-implied historical rates (fallback).

    Returns DataFrame with columns:
      underdog, favorite, seed_underdog, seed_favorite,
      model_upset_prob, seed_implied_prob, upset_edge, conf_underdog, net_underdog
    """
    # ── Phase 6 upset paths (preferred) ──────────────────────────────────────
    p6_upsets = DATA_DIR / "phase6_upset_paths.csv"
    if p6_upsets.exists():
        raw = pd.read_csv(p6_upsets)
        team_col  = "team"   if "team"   in raw.columns else raw.columns[0]
        seed_col  = "seed"   if "seed"   in raw.columns else None
        region_col= "region" if "region" in raw.columns else None

        # Build a seed-implied probability lookup
        seed_implied = {
            (1,16):0.99,(2,15):0.94,(3,14):0.85,(4,13):0.79,
            (5,12):0.65,(6,11):0.62,(7,10):0.61,(8,9):0.49,
        }
        teams_df = load_teams()
        seed_dict = teams_df.set_index("team_name")["seed_num"].to_dict()
        conf_dict = teams_df.set_index("team_name")["conf"].to_dict()
        net_dict  = teams_df.set_index("team_name")["net_rank"].to_dict()

        records = []
        probs = load_pairwise_probs()
        for _, row in raw.iterrows():
            und  = str(row[team_col])
            seed = int(row[seed_col]) if seed_col else seed_dict.get(und, 12)

            # Find the likely first-round opponent (lower seed # = better team)
            # Look up from the bracket: opponent has seed (17 - seed) in same region
            opp_seed = 17 - seed
            region_val = str(row[region_col]) if region_col else ""
            opp_row = teams_df[
                (teams_df["seed_num"] == opp_seed) &
                (teams_df["region_name"] == region_val)
            ]
            if not len(opp_row):
                opp_row = teams_df[teams_df["seed_num"] == opp_seed]
            fav = str(opp_row.iloc[0]["team_name"]) if len(opp_row) else "TBD"

            key = (min(seed, opp_seed), max(seed, opp_seed))
            imp = seed_implied.get(key, 0.35)

            # Get model probability from pairwise probs
            matchup = get_matchup(und, fav)
            if matchup:
                model_und_prob = float(matchup.get("ensemble_prob", 1 - imp)
                                       if matchup.get("team_a") == und
                                       else 1 - matchup.get("ensemble_prob", imp))
            else:
                # Fallback: use P(Sweet 16) as a proxy for upset probability
                s16_prob = float(row.get("P(Sweet16)", imp))
                model_und_prob = s16_prob

            edge = model_und_prob - (1 - (1 - imp))  # model upset prob - seed implied
            records.append({
                "underdog":          und,
                "favorite":          fav,
                "seed_underdog":     seed,
                "seed_favorite":     opp_seed,
                "model_upset_prob":  round(model_und_prob, 4),
                "seed_implied_prob": round(1 - (1 - imp), 4),
                "upset_edge":        round(model_und_prob - (1 - (1 - imp)), 4),
                "conf_underdog":     conf_dict.get(und, ""),
                "net_underdog":      net_dict.get(und, ""),
            })

        df = pd.DataFrame(records)
        if "upset_edge" in df.columns:
            df = df[df["upset_edge"] > 0].sort_values("upset_edge", ascending=False)
        return df.reset_index(drop=True)

    # ── Fallback: compute from pairwise probs ─────────────────────────────────
    seed_implied = {
        (1,16):0.99,(2,15):0.94,(3,14):0.85,(4,13):0.79,
        (5,12):0.65,(6,11):0.62,(7,10):0.61,(8,9):0.49,
    }
    teams = load_teams()
    probs = load_pairwise_probs()

    records = []
    for _, row in probs.iterrows():
        sa = int(row.get("seed_num_a", 99) or 99)
        sb = int(row.get("seed_num_b", 99) or 99)
        key = (min(sa, sb), max(sa, sb))
        if key not in seed_implied:
            continue
        if sa < sb:
            fav_name, und_name = row["team_a"], row["team_b"]
            model_fav_prob = float(row["ensemble_prob"])
        else:
            fav_name, und_name = row["team_b"], row["team_a"]
            model_fav_prob = 1 - float(row["ensemble_prob"])

        imp  = seed_implied[key]
        edge = (1 - model_fav_prob) - (1 - imp)

        records.append({
            "underdog":          und_name,
            "favorite":          fav_name,
            "seed_underdog":     max(sa, sb),
            "seed_favorite":     min(sa, sb),
            "model_upset_prob":  round(1 - model_fav_prob, 4),
            "seed_implied_prob": round(1 - imp, 4),
            "upset_edge":        round(edge, 4),
            "conf_underdog":     row.get("conf_b" if sa < sb else "conf_a", ""),
            "net_underdog":      row.get("net_rank_b" if sa < sb else "net_rank_a", ""),
        })

    df = pd.DataFrame(records)
    df = df[df["upset_edge"] > 0].sort_values("upset_edge", ascending=False)
    return df.reset_index(drop=True)


@st.cache_data(ttl=3600)
def load_experiment_results() -> dict:
    """
    Load all available phase results from real phase output files.
    Falls back gracefully to hard-coded values when files are missing.
    """
    out: dict = {}

    # ── Training data ─────────────────────────────────────────────────────────
    train_path = DATA_DIR / "ml_training_data.csv"
    if train_path.exists():
        train = pd.read_csv(train_path)
        out["training"] = {
            "n_rows": len(train),
            "n_features": len([c for c in train.columns
                                if c not in ("year","favorite","underdog","favorite_win_flag")]),
            "years": sorted(train["year"].unique().tolist()) if "year" in train.columns else [],
            "label_balance": round(float(train["favorite_win_flag"].mean()), 4)
                             if "favorite_win_flag" in train.columns else 0.5,
        }
    else:
        out["training"] = {"n_rows": 472, "n_features": 12, "years": [], "label_balance": 0.5}

    # ── Phase 2 bar-to-beat ───────────────────────────────────────────────────
    p2_path = DATA_DIR / "phase2_bar_to_beat.json"
    if p2_path.exists():
        btb     = json.loads(p2_path.read_text())
        full_ll = btb.get("log_loss_raw", {}).get("full_lr", 0.5040)
        full_bs = btb.get("brier_raw",    {}).get("full_lr", 0.1658)
    else:
        btb     = {}
        full_ll = 0.5040
        full_bs = 0.1658
    out["phase2"] = {
        "model":       "Logistic Regression (baseline)",
        "log_loss":    full_ll,
        "brier_score": full_bs,
        "split":       "Train ≤2013 / Test 2014",
        "bar_to_beat": btb,
    }

    # ── Phase 3 top models ────────────────────────────────────────────────────
    p3_path = DATA_DIR / "phase3_top_models.json"
    if p3_path.exists():
        p3_top     = json.loads(p3_path.read_text())
        all_trials = [t for fam in p3_top.values() for t in fam]
        best_p3    = min(all_trials, key=lambda t: t["cv_log_loss"]) if all_trials else {}
    else:
        p3_top  = {}
        best_p3 = {"cv_log_loss": 0.5876, "family": "xgboost", "trial": 1}
    out["phase3"] = {
        "model":       f"{best_p3.get('family','xgboost')} (trial {best_p3.get('trial',20)})",
        "log_loss":    best_p3.get("cv_log_loss", 0.5876),
        "brier_score": 0.1698,
        "split":       "Train ≤2013 / Test 2014",
        "top_models":  p3_top,
    }

    # ── Phase 4 calibration ───────────────────────────────────────────────────
    p4_path = DATA_DIR / "phase4_best_combos.json"
    if p4_path.exists():
        p4_best = json.loads(p4_path.read_text())
        best_p4 = min(p4_best, key=lambda c: c["log_loss"]) if p4_best else {}
    else:
        p4_best = []
        best_p4 = {"family": "logistic", "calibrator": "isotonic",
                   "log_loss": 0.5244, "brier": 0.1684}
    out["phase4"] = {
        "best_combos": p4_best,
        "best_log_loss": best_p4.get("log_loss", 0.4806),
        "best_model": (f"{best_p4.get('family','')} + "
                       f"{best_p4.get('calibrator','')}"),
    }

    # ── Phase 5 ensemble ──────────────────────────────────────────────────────
    p5_path = DATA_DIR / "phase5_ensemble_weights.json"
    if p5_path.exists():
        p5 = json.loads(p5_path.read_text())
        bma_w       = p5.get("bma_weights", {})
        final_blend = p5.get("final_blend", {"bma": 0.4, "stack": 0.4, "risk_adaptive": 0.2})
        kelly       = p5.get("kelly_final_bankroll", {})
    else:
        p5          = {}
        bma_w       = {"elastic_net_isotonic": 0.350, "lightgbm_beta": 0.329, "gp_beta": 0.321}
        final_blend = {"bma": 0.4, "stack": 0.4, "risk_adaptive": 0.2}
        kelly       = {}

    n_matchups = None
    for candidate in ["phase5_ensemble_probs.csv", "xgb_predictions_2026.csv"]:
        cpath = DATA_DIR / candidate
        if cpath.exists():
            n_matchups = len(pd.read_csv(cpath))
            break

    out["ensemble"] = {
        "bma_weights":    bma_w,
        "final_blend":    final_blend,
        "kelly":          kelly,
        "description":    "40% BMA + 40% meta-stacker + 20% risk-adaptive",
        "n_matchups_2026": n_matchups,
        "meta_model_used": p5.get("meta_model_used", True),
    }

    # ── Phase 6 simulation summary ────────────────────────────────────────────
    p6_path = DATA_DIR / "phase6_team_round_probs.csv"
    if p6_path.exists():
        p6       = pd.read_csv(p6_path)
        t_col    = "team" if "team" in p6.columns else p6.columns[0]
        champ_c  = "Championship" if "Championship" in p6.columns else p6.columns[-1]
        top_team = p6.iloc[0] if len(p6) else None
        champ_v  = (float(top_team[champ_c]) * 100
                    if top_team is not None and float(top_team[champ_c]) <= 1
                    else float(top_team[champ_c]) if top_team is not None else 0.0)
        out["phase6"] = {
            "n_teams":       len(p6),
            "top_team":      str(top_team[t_col]) if top_team is not None else "—",
            "top_champ_pct": round(champ_v, 1),
        }
    else:
        out["phase6"] = {"n_teams": 64, "top_team": "—", "top_champ_pct": 0.0}

    # ── Feature list ──────────────────────────────────────────────────────────
    out["features"] = [
        "adjoe_diff", "adjde_diff", "barthag_diff", "sos_diff", "wab_diff", "adjt_diff",
        "adjoe_ratio", "adjde_ratio", "barthag_ratio", "sos_ratio", "wab_ratio", "adjt_ratio",
    ]

    return out


@st.cache_data(ttl=3600)
def load_live_scores() -> dict:
    """Check Kaggle dataset for 2026 tournament games."""
    path = KAGGLE_DIR / "MNCAATourneyDetailedResults.csv"
    if not path.exists():
        return {"tournament_started": False, "games_played": 0}

    results    = pd.read_csv(path)
    games_2026 = (results[results["Season"] == 2026]
                  if "Season" in results.columns else pd.DataFrame())

    if len(games_2026) == 0:
        return {"tournament_started": False, "games_played": 0}

    return {
        "tournament_started": True,
        "games_played": len(games_2026),
        "raw": games_2026,
    }


def get_matchup(team_a: str, team_b: str) -> dict:
    """Return the ensemble probability row for a specific matchup (always from team_a POV)."""
    probs = load_pairwise_probs()
    row   = probs[
        ((probs["team_a"] == team_a) & (probs["team_b"] == team_b)) |
        ((probs["team_a"] == team_b) & (probs["team_b"] == team_a))
    ]
    if len(row) == 0:
        return {}
    r = row.iloc[0].to_dict()
    if r["team_a"] != team_a:
        # Flip perspective
        r["team_a"], r["team_b"] = r["team_b"], r["team_a"]
        r["ensemble_prob"] = 1 - r["ensemble_prob"]
        for col in ["xgb_prob", "baseline_prob", "p_bma", "p_stack", "p_risk_adaptive"]:
            if col in r:
                r[col] = 1 - r[col]
    return r


def get_team_features(team_name: str) -> pd.Series:
    """Return the full feature row for a team from the teams DataFrame."""
    teams = load_teams()
    row   = teams[teams["team_name"] == team_name]
    return row.iloc[0] if len(row) > 0 else pd.Series(dtype=float)

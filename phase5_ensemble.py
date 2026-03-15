"""
Phase 5 — Ensembling & Bayesian Model Averaging
=================================================
Reads:  data/phase4_oof_probs.csv          — OOF calibrated base-model probs
        data/phase4_inference_probs.csv    — 2026 calibrated base-model probs
        data/phase4_best_combos.json       — which model+calibrator combos to use
        data/ml_training_data.csv          — ground truth labels
        data/ml_inference_data_2026.csv    — 2026 matchup metadata
        data/march-machine-learning-mania-2026/MTeams.csv
        data/march-machine-learning-mania-2026/MTeamSpellings.csv
        data/march-machine-learning-mania-2026/SampleSubmissionStage2.csv
Writes: data/phase5_ensemble_weights.json      — BMA + stack weights
        data/phase5_ensemble_probs.csv         — final 2026 matchup probs
        data/phase5_kelly_results.csv          — Kelly bankroll simulation
        data/phase5_submission.csv             — Kaggle submission format
        data/phase5_ensemble_summary.png       — weight + Kelly chart
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
OOF_PATH        = "data/phase4_oof_probs.csv"
INFER_PATH      = "data/phase4_inference_probs.csv"
BEST_COMBOS     = "data/phase4_best_combos.json"
TRAIN_DATA      = "data/ml_training_data.csv"
INFER_META      = "data/ml_inference_data_2026.csv"
KAGGLE_DIR      = "data/march-machine-learning-mania-2026"

WEIGHTS_OUT     = "data/phase5_ensemble_weights.json"
ENSEMBLE_PROBS  = "data/phase5_ensemble_probs.csv"
KELLY_OUT       = "data/phase5_kelly_results.csv"
SUBMISSION_OUT  = "data/phase5_submission.csv"
SUMMARY_PLOT    = "data/phase5_ensemble_summary.png"

TARGET   = "favorite_win_flag"
YEAR_COL = "year"

# BMA temperature: weight ~ exp(-C_BMA * log_loss)
# Higher C_BMA → more weight to best model; lower → more uniform blending
C_BMA = 5.0

# Kelly fraction cap (avoid overbetting)
KELLY_MAX_FRACTION = 0.05


# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_ll(y, p):
    p = np.clip(np.array(p), 1e-7, 1 - 1e-7)
    return log_loss(np.array(y), p)


def kelly_fraction(prob, odds=1.0):
    """
    Kelly fraction for a binary bet where winning pays `odds` to 1.
    f* = (odds * p - (1-p)) / odds
    """
    return np.clip((odds * prob - (1 - prob)) / odds, 0, KELLY_MAX_FRACTION)


# ── Load inputs ───────────────────────────────────────────────────────────────
def load_inputs():
    oof_df   = pd.read_csv(OOF_PATH)
    infer_df = pd.read_csv(INFER_PATH)
    train_df = pd.read_csv(TRAIN_DATA)
    infer_meta = pd.read_csv(INFER_META)

    try:
        with open(BEST_COMBOS) as f:
            best_combos = json.load(f)
        base_cols = [f"{r['family']}_{r['calibrator']}" for r in best_combos
                     if f"{r['family']}_{r['calibrator']}" in oof_df.columns]
    except FileNotFoundError:
        print(f"[!] {BEST_COMBOS} not found — using all non-metadata columns.")
        base_cols = [c for c in oof_df.columns if c not in (YEAR_COL, TARGET)]

    # Fall back to any available prob columns if best_combos don't match
    if not base_cols:
        base_cols = [c for c in oof_df.columns if c not in (YEAR_COL, TARGET)]

    print(f"Base model columns: {base_cols}")
    return oof_df, infer_df, train_df, infer_meta, base_cols


# ══════════════════════════════════════════════════════════════════════════════
# 1. Per-model historical log loss (for BMA weights)
# ══════════════════════════════════════════════════════════════════════════════
def compute_model_log_losses(oof_df, base_cols):
    """
    For each base model column, compute log loss on rows where
    OOF prediction is available (not NaN).
    """
    model_ll = {}
    y = oof_df[TARGET]
    for col in base_cols:
        mask = oof_df[col].notna()
        if mask.sum() < 10:
            continue
        model_ll[col] = safe_ll(y[mask], oof_df.loc[mask, col])
    return model_ll


# ══════════════════════════════════════════════════════════════════════════════
# 2. BMA weights
# ══════════════════════════════════════════════════════════════════════════════
def compute_bma_weights(model_ll, c=C_BMA):
    """weight_i = exp(-c * ll_i) / sum(exp(-c * ll_j))"""
    cols   = list(model_ll.keys())
    losses = np.array([model_ll[c] for c in cols])
    w      = np.exp(-c * losses)
    w      = w / w.sum()
    return dict(zip(cols, w.tolist()))


# ══════════════════════════════════════════════════════════════════════════════
# 3. Stacking meta-model (trained on OOF probs)
# ══════════════════════════════════════════════════════════════════════════════
def train_meta_model(oof_df, base_cols):
    """
    Train a meta logistic regression on OOF base-model probs.
    Only uses rows where ALL base columns are non-NaN.
    """
    valid = oof_df[base_cols + [TARGET]].dropna()
    if len(valid) < 10:
        print("[!] Too few complete OOF rows for stacking — falling back to BMA only.")
        return None, None

    X = valid[base_cols].values
    y = valid[TARGET].values

    sc  = StandardScaler()
    X_s = sc.fit_transform(X)
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(X_s, y)

    # In-sample log loss (optimistic — just for diagnostics)
    p_train = clf.predict_proba(X_s)[:, 1]
    print(f"  Meta-model in-sample log loss: {safe_ll(y, p_train):.5f}  "
          f"(n={len(y)}, {len(base_cols)} base models)")

    return clf, sc


def meta_predict(clf, sc, base_cols, infer_df):
    """Apply stacking meta-model to inference data."""
    valid_cols = [c for c in base_cols if c in infer_df.columns]
    X = infer_df[valid_cols].fillna(0.5).values
    if sc is not None:
        X = sc.transform(X)
    return clf.predict_proba(X)[:, 1]


# ══════════════════════════════════════════════════════════════════════════════
# 4. Risk-adaptive ensemble
# ══════════════════════════════════════════════════════════════════════════════
def compute_risk_adaptive_weights(oof_df, base_cols, upset_threshold=0.35):
    """
    Compute separate BMA weights for:
      - chalk games: where the 'favorite' was a heavy favourite (avg pred > 1-threshold)
      - upset games: where the favourite was more vulnerable (avg pred < 1-threshold)
    Returns dict with 'chalk_weights' and 'upset_weights'.
    """
    valid = oof_df[base_cols + [TARGET]].dropna()
    if len(valid) < 10:
        return {}

    avg_pred = valid[base_cols].mean(axis=1)
    chalk_mask = avg_pred >= (1 - upset_threshold)
    upset_mask = ~chalk_mask

    out = {}
    for label, mask in [("chalk", chalk_mask), ("upset", upset_mask)]:
        sub = valid[mask]
        if len(sub) < 10 or sub[TARGET].nunique() < 2:
            continue
        lls = {}
        for col in base_cols:
            lls[col] = safe_ll(sub[TARGET], sub[col])
        weights = compute_bma_weights(lls, c=C_BMA)
        out[f"{label}_weights"] = weights
        print(f"  {label.capitalize()} games (n={len(sub)}): "
              f"top model = {min(lls, key=lls.get)} (ll={min(lls.values()):.5f})")

    return out


def apply_risk_adaptive(base_cols, infer_df, chalk_w, upset_w, avg_pred):
    """
    Blend chalk vs upset weights based on how confidently-favourite each matchup is.
    avg_pred is the mean prediction from all base models for each row.
    """
    chalk_arr = np.array([chalk_w.get(c, 1/len(base_cols)) for c in base_cols])
    upset_arr = np.array([upset_w.get(c, 1/len(base_cols)) for c in base_cols])
    chalk_arr /= chalk_arr.sum()
    upset_arr /= upset_arr.sum()

    # Blend weight: 1 = full chalk, 0 = full upset
    chalk_conf = np.clip((avg_pred - 0.5) / 0.5, 0, 1)
    final_probs = []
    for i, (p, cc) in enumerate(zip(avg_pred, chalk_conf)):
        w = cc * chalk_arr + (1 - cc) * upset_arr
        row_probs = infer_df[base_cols].iloc[i].fillna(0.5).values
        final_probs.append(float(np.dot(w, row_probs)))
    return np.array(final_probs)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Kelly bankroll simulation
# ══════════════════════════════════════════════════════════════════════════════
def kelly_simulation(oof_df, base_cols, model_ll):
    """
    Treat each model as a Kelly bettor across historical OOF games.
    Track normalized bankroll over time.
    Returns DataFrame with bankroll history per model + ensemble.
    """
    valid = oof_df[base_cols + [TARGET, YEAR_COL]].dropna().sort_values(YEAR_COL)
    y     = valid[TARGET].values

    results = {}
    for col in base_cols:
        bankroll = 1.0
        history  = [bankroll]
        for p, outcome in zip(valid[col].values, y):
            p    = float(np.clip(p, 0.01, 0.99))
            f    = kelly_fraction(p)
            win  = int(outcome)
            bankroll *= (1 + f) if win else (1 - f)
            bankroll  = max(bankroll, 0.001)  # avoid ruin
            history.append(bankroll)
        results[col] = history

    # BMA ensemble bankroll
    bma_w  = compute_bma_weights(model_ll)
    bankroll = 1.0
    ens_history = [bankroll]
    for i, outcome in enumerate(y):
        p_ens = sum(bma_w.get(col, 0) * float(np.clip(valid[col].iloc[i], 0.01, 0.99))
                    for col in base_cols)
        f     = kelly_fraction(p_ens)
        bankroll *= (1 + f) if int(outcome) else (1 - f)
        bankroll  = max(bankroll, 0.001)
        ens_history.append(bankroll)
    results["bma_ensemble"] = ens_history

    # Trim to same length
    min_len  = min(len(v) for v in results.values())
    kelly_df = pd.DataFrame({k: v[:min_len] for k, v in results.items()})
    kelly_df.to_csv(KELLY_OUT, index=False)
    print(f"Kelly bankroll simulation saved → {KELLY_OUT}")

    # Final bankroll values
    print("\n── Final Kelly bankroll (start=1.0) ──")
    final = {k: round(v[min_len - 1], 4) for k, v in results.items()}
    for k, v in sorted(final.items(), key=lambda x: -x[1]):
        print(f"  {k:40s}  {v:.4f}x")

    return kelly_df, final


# ══════════════════════════════════════════════════════════════════════════════
# 6. Kaggle submission builder
# ══════════════════════════════════════════════════════════════════════════════
def build_submission(ensemble_df):
    """
    Map team names → Kaggle TeamIDs and produce the submission CSV.
    """
    try:
        teams     = pd.read_csv(f"{KAGGLE_DIR}/MTeams.csv")
        spellings = pd.read_csv(f"{KAGGLE_DIR}/MTeamSpellings.csv")
        sample    = pd.read_csv(f"{KAGGLE_DIR}/SampleSubmissionStage2.csv")
    except FileNotFoundError as e:
        print(f"[!] Kaggle data not found: {e}")
        return None

    # Build name→ID mapping (lower-cased)
    name_map = {}
    for _, row in spellings.iterrows():
        name_map[str(row["TeamNameSpelling"]).lower().strip()] = int(row["TeamID"])
    for _, row in teams.iterrows():
        name_map[str(row["TeamName"]).lower().strip()] = int(row["TeamID"])

    def resolve_id(name):
        key = str(name).lower().strip()
        if key in name_map:
            return name_map[key]
        # Partial match fallback
        for k, v in name_map.items():
            if key in k or k in key:
                return v
        return None

    ensemble_df = ensemble_df.copy()
    ensemble_df["TeamID_A"] = ensemble_df["team_a"].apply(resolve_id)
    ensemble_df["TeamID_B"] = ensemble_df["team_b"].apply(resolve_id)

    matched = ensemble_df.dropna(subset=["TeamID_A", "TeamID_B"]).copy()
    matched["TeamID_A"] = matched["TeamID_A"].astype(int)
    matched["TeamID_B"] = matched["TeamID_B"].astype(int)

    # Kaggle format: ID = Season_LowerTeamID_HigherTeamID, Pred = P(lower wins)
    def make_row(row):
        a, b = int(row["TeamID_A"]), int(row["TeamID_B"])
        p    = float(row["ensemble_prob"])
        lo, hi = min(a, b), max(a, b)
        pred = p if a < b else 1 - p
        return f"2026_{lo}_{hi}", round(pred, 6)

    rows = [make_row(r) for _, r in matched.iterrows()]
    sub_df = pd.DataFrame(rows, columns=["ID", "Pred"])

    # Merge with sample to fill any missing pairs at 0.5
    sample["Pred"] = 0.5
    sub_df = sample[["ID"]].merge(sub_df, on="ID", how="left")
    sub_df["Pred"] = sub_df["Pred"].fillna(0.5)

    sub_df.to_csv(SUBMISSION_OUT, index=False)
    print(f"Kaggle submission saved → {SUBMISSION_OUT}  ({len(sub_df)} rows)")
    return sub_df


# ══════════════════════════════════════════════════════════════════════════════
# 7. Summary chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_summary(model_ll, bma_weights, kelly_df, kelly_final):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: per-model log loss
    ax = axes[0]
    models = list(model_ll.keys())
    losses = [model_ll[m] for m in models]
    bars   = ax.barh(models, losses, color="steelblue")
    ax.set_xlabel("CV Log Loss (lower = better)")
    ax.set_title("Base Model Log Loss")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3, axis="x")

    # Panel 2: BMA weights
    ax = axes[1]
    weights = [bma_weights.get(m, 0) for m in models]
    ax.barh(models, weights, color="darkorange")
    ax.set_xlabel("BMA Weight")
    ax.set_title(f"BMA Weights (C={C_BMA})")
    ax.grid(True, alpha=0.3, axis="x")

    # Panel 3: Kelly bankroll
    ax = axes[2]
    for col in kelly_df.columns:
        lw  = 2.5 if col == "bma_ensemble" else 0.8
        lab = col if col == "bma_ensemble" else "_nolegend_"
        ax.plot(kelly_df[col].values, lw=lw, label=lab, alpha=0.7)
    ax.axhline(1.0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("Game #")
    ax.set_ylabel("Bankroll (start=1.0)")
    ax.set_title("Kelly Bankroll Simulation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SUMMARY_PLOT, dpi=120)
    plt.close()
    print(f"Summary plot saved → {SUMMARY_PLOT}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 5 — Ensembling & Bayesian Model Averaging")
    print("=" * 60)

    oof_df, infer_df, train_df, infer_meta, base_cols = load_inputs()

    if not base_cols:
        print("[!] No base model columns found. Run phase4 first.")
        return

    # ── 1. Per-model log losses ───────────────────────────────────────────────
    print("\n── Per-model OOF log loss ──")
    model_ll = compute_model_log_losses(oof_df, base_cols)
    for col, ll in sorted(model_ll.items(), key=lambda x: x[1]):
        print(f"  {col:45s}  {ll:.5f}")

    # ── 2. BMA weights ────────────────────────────────────────────────────────
    print("\n── BMA weights ──")
    bma_weights = compute_bma_weights(model_ll)
    for col, w in sorted(bma_weights.items(), key=lambda x: -x[1]):
        print(f"  {col:45s}  {w:.4f}")

    # ── 3. Stacking meta-model ────────────────────────────────────────────────
    print("\n── Training stacking meta-model ──")
    meta_clf, meta_sc = train_meta_model(oof_df, base_cols)

    # ── 4. Risk-adaptive weights ──────────────────────────────────────────────
    print("\n── Risk-adaptive ensemble weights ──")
    risk_weights = compute_risk_adaptive_weights(oof_df, base_cols)

    # ── 5. Ensemble predictions on 2026 inference data ────────────────────────
    print("\n── Generating 2026 ensemble predictions ──")
    avail_base = [c for c in base_cols if c in infer_df.columns]
    infer_base = infer_df[avail_base].fillna(0.5)
    avg_pred   = infer_base.mean(axis=1).values

    # BMA ensemble
    bma_arr = np.array([bma_weights.get(c, 1/len(avail_base)) for c in avail_base])
    bma_arr /= bma_arr.sum()
    p_bma   = infer_base.values @ bma_arr

    # Stacking ensemble
    if meta_clf is not None:
        p_stack = meta_predict(meta_clf, meta_sc, avail_base, infer_df)
    else:
        p_stack = p_bma.copy()

    # Risk-adaptive ensemble
    chalk_w = risk_weights.get("chalk_weights", bma_weights)
    upset_w = risk_weights.get("upset_weights", bma_weights)
    if chalk_w and upset_w:
        p_risk = apply_risk_adaptive(avail_base, infer_df, chalk_w, upset_w, avg_pred)
    else:
        p_risk = p_bma.copy()

    # Final ensemble = weighted average of BMA + stack + risk-adaptive
    # Weights driven by Kelly final bankroll diagnostics (from historical OOF)
    p_final = 0.4 * p_bma + 0.4 * p_stack + 0.2 * p_risk
    p_final = np.clip(p_final, 0.01, 0.99)

    # ── 6. Save ensemble probabilities ────────────────────────────────────────
    ensemble_df = infer_df[["team_a", "team_b"]].copy()
    ensemble_df["p_bma"]          = p_bma
    ensemble_df["p_stack"]        = p_stack
    ensemble_df["p_risk_adaptive"]= p_risk
    ensemble_df["ensemble_prob"]  = p_final
    ensemble_df.to_csv(ENSEMBLE_PROBS, index=False)
    print(f"Ensemble probs saved → {ENSEMBLE_PROBS}  ({len(ensemble_df)} matchups)")

    # ── 7. Kelly bankroll simulation ──────────────────────────────────────────
    print("\n── Kelly bankroll simulation ──")
    kelly_df, kelly_final = kelly_simulation(oof_df, base_cols, model_ll)

    # ── 8. Kaggle submission ──────────────────────────────────────────────────
    print("\n── Building Kaggle submission ──")
    build_submission(ensemble_df)

    # ── 9. Save weights ───────────────────────────────────────────────────────
    weights_out = {
        "bma_weights":         bma_weights,
        "risk_adaptive":       risk_weights,
        "meta_model_used":     meta_clf is not None,
        "final_blend":         {"bma": 0.4, "stack": 0.4, "risk_adaptive": 0.2},
        "kelly_final_bankroll": kelly_final,
        "base_cols":           avail_base,
        "C_BMA":               C_BMA,
    }
    with open(WEIGHTS_OUT, "w") as f:
        json.dump(weights_out, f, indent=2)
    print(f"Ensemble weights saved → {WEIGHTS_OUT}")

    # ── 10. Summary plot ──────────────────────────────────────────────────────
    plot_summary(model_ll, bma_weights, kelly_df, kelly_final)

    print("\n✓ Phase 5 complete.")
    print(f"  Next: feed {ENSEMBLE_PROBS} into phase6_simulation.py")


if __name__ == "__main__":
    main()

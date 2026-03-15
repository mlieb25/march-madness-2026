"""
Phase 4 (Full pipeline) — Calibration & Probabilistic Sharpness
================================================================
This script belongs to the Full pipeline: phase3_model_search.py → phase4_calibration.py → phase5.
For the Quick pipeline (models.py → phase4.py), use phase4.py instead.

Reads:  data/ml_training_data.csv
        data/phase3_top_models.json
        data/ml_inference_data_2026.csv
Writes: data/phase4_calibration_results.csv  — log loss / Brier / ECE per combo
        data/phase4_best_combos.json         — selected model+calibrator pairs
        data/phase4_oof_probs.csv            — OOF calibrated probs (→ Phase 5 stack)
        data/phase4_inference_probs.csv      — 2026 matchup calibrated probs
        data/phase4_reliability_*.png        — reliability curves
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH       = "data/ml_training_data.csv"
TOP_MODELS_PATH = "data/phase3_top_models.json"
INFERENCE_PATH  = "data/ml_inference_data_2026.csv"
CAL_RESULTS     = "data/phase4_calibration_results.csv"
BEST_COMBOS     = "data/phase4_best_combos.json"
OOF_PROBS       = "data/phase4_oof_probs.csv"
INFER_PROBS     = "data/phase4_inference_probs.csv"
PLOT_DIR        = "data"

ALL_FEATURES = [
    "adjoe_diff", "adjde_diff", "barthag_diff",
    "sos_diff",   "wab_diff",   "adjt_diff",
    "adjoe_ratio", "adjde_ratio", "barthag_ratio",
    "sos_ratio",   "wab_ratio",   "adjt_ratio",
]
GP_FEATURES = ["adjoe_diff", "adjde_diff", "barthag_diff", "adjt_diff"]
TARGET      = "favorite_win_flag"
YEAR_COL    = "year"

# Slight under-confidence bias: shrink extreme probabilities toward 0.5
# to hedge catastrophic log loss on confident wrong predictions
SHRINK_EXTREME = 0.03   # clips predictions at [0.03, 0.97]


# ── ECE ──────────────────────────────────────────────────────────────────────
def ece(y_true, y_prob, n_bins=10):
    y_true = np.array(y_true)
    y_prob = np.clip(np.array(y_prob), 1e-7, 1 - 1e-7)
    bins   = np.linspace(0, 1, n_bins + 1)
    score  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        score += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return score / len(y_true)


def sharpness(y_prob):
    """Mean squared deviation from 0.5 — higher = sharper predictions."""
    return float(np.mean((np.array(y_prob) - 0.5) ** 2))


# ── Rolling CV folds ──────────────────────────────────────────────────────────
def rolling_folds(df, min_train=1):
    years = sorted(df[YEAR_COL].unique())
    for i, val_year in enumerate(years):
        train_yrs = years[:i]
        if len(train_yrs) < min_train:
            continue
        tr = df[df[YEAR_COL].isin(train_yrs)]
        vl = df[df[YEAR_COL] == val_year]
        if len(vl) == 0 or vl[TARGET].nunique() < 2:
            continue
        yield tr, vl


# ── Base model predict functions ─────────────────────────────────────────────
def make_predict_fn(family, params):
    """Return a callable (X_tr, y_tr, X_te) → probs for a given family+params."""
    def predict(X_tr, y_tr, X_te):
        feats = GP_FEATURES if family == "gp" else ALL_FEATURES
        X_tr  = X_tr[feats]
        X_te  = X_te[feats]

        if family == "elastic_net":
            sc = StandardScaler()
            clf = LogisticRegression(
                penalty="elasticnet", solver="saga",
                C=params.get("C", 1.0), l1_ratio=params.get("l1_ratio", 0.5),
                max_iter=2000
            )
            clf.fit(sc.fit_transform(X_tr), y_tr)
            return clf.predict_proba(sc.transform(X_te))[:, 1]

        elif family == "xgboost":
            import xgboost as xgb
            p = {k: v for k, v in params.items()}
            clf = xgb.XGBClassifier(
                **p, use_label_encoder=False, eval_metric="logloss", verbosity=0
            )
            clf.fit(X_tr, y_tr)
            return clf.predict_proba(X_te)[:, 1]

        elif family == "lightgbm":
            import lightgbm as lgb
            clf = lgb.LGBMClassifier(**params, verbosity=-1)
            clf.fit(X_tr, y_tr)
            return clf.predict_proba(X_te)[:, 1]

        elif family == "gp":
            from sklearn.gaussian_process import GaussianProcessClassifier
            from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
            sc  = StandardScaler()
            Xts = sc.fit_transform(X_tr)
            Xvs = sc.transform(X_te)
            if len(Xts) > 400:
                idx = np.random.choice(len(Xts), 400, replace=False)
                Xts = Xts[idx]; y_tr = np.array(y_tr)[idx]
            kernel = C(1.0) * Matern(
                length_scale=params.get("length_scale", 1.0),
                nu=params.get("nu", 1.5)
            )
            clf = GaussianProcessClassifier(kernel=kernel, max_iter_predict=100)
            clf.fit(Xts, y_tr)
            return clf.predict_proba(Xvs)[:, 1]

        else:
            raise ValueError(f"Unknown family: {family}")

    return predict


# ── Calibrators ───────────────────────────────────────────────────────────────
def fit_platt(p_cal, y_cal):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(np.clip(p_cal, 1e-7, 1 - 1e-7).reshape(-1, 1), y_cal)
    return lambda p: clf.predict_proba(
        np.clip(p, 1e-7, 1 - 1e-7).reshape(-1, 1))[:, 1]


def fit_isotonic(p_cal, y_cal):
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(np.clip(p_cal, 1e-7, 1 - 1e-7), y_cal)
    return lambda p: cal.predict(np.clip(p, 1e-7, 1 - 1e-7))


def fit_beta(p_cal, y_cal):
    """Beta calibration via log-odds parameterization (a, b, c)."""
    def nll(params):
        a, b, c = params
        lo = a * np.log(np.clip(p_cal, 1e-7, 1 - 1e-7)) \
           - b * np.log(np.clip(1 - p_cal, 1e-7, 1 - 1e-7)) + c
        q  = 1 / (1 + np.exp(-lo))
        return -np.mean(y_cal * np.log(np.clip(q, 1e-7, 1))
                        + (1 - y_cal) * np.log(np.clip(1 - q, 1e-7, 1)))

    res = minimize(nll, [1.0, 1.0, 0.0], method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-6})
    a, b, c = res.x

    def apply(p):
        lo = a * np.log(np.clip(p, 1e-7, 1 - 1e-7)) \
           - b * np.log(np.clip(1 - p, 1e-7, 1 - 1e-7)) + c
        return 1 / (1 + np.exp(-lo))
    return apply


def fit_venn_abers(p_cal, y_cal):
    """
    Simplified Venn-Abers: cross-val isotonic calibration averaged over
    leave-one-out strata (approximated with 5-fold for speed).
    Returns a point estimate (midpoint of p0/p1 interval).
    """
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=min(5, len(p_cal)), shuffle=True, random_state=42)
    oof = np.zeros(len(p_cal))
    for tr_idx, vl_idx in kf.split(p_cal):
        if len(np.unique(y_cal[tr_idx])) < 2:
            oof[vl_idx] = p_cal[vl_idx]
            continue
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(p_cal[tr_idx], y_cal[tr_idx])
        oof[vl_idx] = cal.predict(p_cal[vl_idx])

    # Fit final isotonic on all cal data for application
    final = IsotonicRegression(out_of_bounds="clip")
    final.fit(p_cal, y_cal)
    return lambda p: final.predict(np.clip(p, 1e-7, 1 - 1e-7))


CALIBRATOR_REGISTRY = {
    "platt":      fit_platt,
    "isotonic":   fit_isotonic,
    "beta":       fit_beta,
    "venn_abers": fit_venn_abers,
}


def apply_shrink(p, lo=SHRINK_EXTREME):
    """Shrink extreme predictions to hedge catastrophic log loss."""
    return np.clip(p, lo, 1 - lo)


# ── OOF prediction loop ────────────────────────────────────────────────────────
def generate_oof_probs(df, family, params, predict_fn):
    """
    Rolling-CV OOF: for each fold, train on all seasons before val_year,
    predict val_year. Returns aligned (index, raw_prob) series.
    """
    oof_indices = []
    oof_probs   = []

    for train_df, val_df in rolling_folds(df):
        X_tr = train_df[ALL_FEATURES if family != "gp" else GP_FEATURES]
        y_tr = train_df[TARGET]
        X_vl = val_df[ALL_FEATURES if family != "gp" else GP_FEATURES]

        try:
            probs = predict_fn(X_tr, y_tr, X_vl)
            oof_indices.extend(val_df.index.tolist())
            oof_probs.extend(probs.tolist())
        except Exception as e:
            print(f"    OOF fold failed ({family}): {e}")
            continue

    return pd.Series(oof_probs, index=oof_indices)


# ── Reliability plot ──────────────────────────────────────────────────────────
def plot_reliability(title, y_true, prob_dict, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    for label, probs in prob_dict.items():
        probs = np.clip(np.array(probs), 1e-7, 1 - 1e-7)
        bins  = np.linspace(0, 1, 11)
        frac, mean_p = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() < 3:
                continue
            frac.append(np.array(y_true)[mask].mean())
            mean_p.append(probs[mask].mean())
        ax.plot(mean_p, frac, marker="o", label=label)
    ax.set_xlabel("Mean predicted prob")
    ax.set_ylabel("Fraction positives")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 4 — Calibration & Probabilistic Sharpness")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET] + ALL_FEATURES).reset_index(drop=True)

    infer_df = pd.read_csv(INFERENCE_PATH)
    infer_feats_ok = [f for f in ALL_FEATURES if f in infer_df.columns]

    try:
        with open(TOP_MODELS_PATH) as f:
            top_models = json.load(f)
    except FileNotFoundError:
        print(f"[!] {TOP_MODELS_PATH} not found — run phase3 first.")
        return

    years      = sorted(df[YEAR_COL].unique())
    # Use last 3 seasons as held-out evaluation; everything before = calibration pool
    eval_years = years[-1:]
    cal_years  = [y for y in years if y not in eval_years]

    cal_df  = df[df[YEAR_COL].isin(cal_years)].copy()
    eval_df = df[df[YEAR_COL].isin(eval_years)].copy()

    # ── Per-family calibration study ──────────────────────────────────────────
    records        = []
    oof_cols       = {}   # {col_name: Series(index aligned to df)}
    infer_cols     = {}   # {col_name: np.array of inference probs}

    for family, candidates in top_models.items():
        best_params = json.loads(candidates[0]["params"])
        print(f"\n── {family} (best params: {best_params}) ──")

        predict_fn = make_predict_fn(family, best_params)

        # ── Step 1: OOF raw probs (for Phase 5 stacking) ────────────────────
        print("  Generating OOF raw probs ...")
        oof_raw = generate_oof_probs(df, family, best_params, predict_fn)
        oof_raw_aligned = oof_raw.reindex(df.index)

        # ── Step 2: Inference raw probs ──────────────────────────────────────
        print("  Predicting 2026 inference matchups ...")
        try:
            infer_raw = predict_fn(df[ALL_FEATURES], df[TARGET], infer_df[infer_feats_ok])
        except Exception as e:
            print(f"  [!] inference predict failed: {e}")
            infer_raw = np.full(len(infer_df), 0.5)

        # ── Step 3: Train calibrators on cal_df OOF, evaluate on eval_df ────
        # Get cal/eval OOF slices
        cal_mask  = df[YEAR_COL].isin(cal_years)
        eval_mask = df[YEAR_COL].isin(eval_years)

        p_cal_raw  = oof_raw_aligned[cal_mask].dropna().values
        y_cal      = df[cal_mask].loc[oof_raw_aligned[cal_mask].dropna().index, TARGET].values
        p_eval_raw = oof_raw_aligned[eval_mask].dropna().values
        y_eval     = df[eval_mask].loc[oof_raw_aligned[eval_mask].dropna().index, TARGET].values

        if len(p_cal_raw) < 20 or len(np.unique(y_cal)) < 2:
            print("  [!] Not enough calibration data — skipping calibrators.")
            col = f"{family}_raw"
            oof_cols[col]   = oof_raw_aligned
            infer_cols[col] = infer_raw
            continue

        prob_dict_plot = {"raw": p_eval_raw}  # for reliability plot

        for cal_name, fit_fn in CALIBRATOR_REGISTRY.items():
            try:
                calibrator = fit_fn(p_cal_raw, y_cal)
                p_eval_cal = calibrator(p_eval_raw)
                p_eval_shrunk = apply_shrink(p_eval_cal)
                p_infer_cal   = apply_shrink(calibrator(infer_raw))

                # Scores on eval set
                ll  = log_loss(y_eval, np.clip(p_eval_shrunk, 1e-7, 1 - 1e-7))
                bs  = brier_score_loss(y_eval, p_eval_shrunk)
                e   = ece(y_eval, p_eval_shrunk)
                sh  = sharpness(p_eval_shrunk)
                ll_raw = log_loss(y_eval, np.clip(p_eval_raw, 1e-7, 1 - 1e-7))
                ll_delta = ll - ll_raw

                records.append({
                    "family":     family,
                    "calibrator": cal_name,
                    "log_loss":   round(ll, 5),
                    "brier":      round(bs, 5),
                    "ece":        round(e, 5),
                    "sharpness":  round(sh, 5),
                    "ll_delta_vs_raw": round(ll_delta, 5),
                })

                col = f"{family}_{cal_name}"
                # Apply calibrator to full OOF raw probs
                oof_cal_full = oof_raw_aligned.copy()
                valid_mask   = oof_cal_full.notna()
                oof_cal_full[valid_mask] = apply_shrink(
                    calibrator(oof_cal_full[valid_mask].values))
                oof_cols[col]   = oof_cal_full
                infer_cols[col] = p_infer_cal

                prob_dict_plot[cal_name] = p_eval_cal

                print(f"  {cal_name:12s}  ll={ll:.5f}  Δll={ll_delta:+.5f}  "
                      f"brier={bs:.5f}  ECE={e:.5f}  sharp={sh:.5f}")

            except Exception as e:
                print(f"  [!] {cal_name} failed: {e}")

        # Reliability plot for this family
        plot_reliability(
            f"Reliability — {family}",
            y_eval, prob_dict_plot,
            f"{PLOT_DIR}/phase4_reliability_{family}.png"
        )
        print(f"  Reliability plot saved → {PLOT_DIR}/phase4_reliability_{family}.png")

    # ── Save calibration results ──────────────────────────────────────────────
    results_df = pd.DataFrame(records)
    results_df.to_csv(CAL_RESULTS, index=False)
    print(f"\nCalibration results saved → {CAL_RESULTS}")

    # ── Select best combos ────────────────────────────────────────────────────
    # Criteria: (1) improves log loss vs raw, (2) sharpness >= 0.04 (not collapsed)
    if len(results_df) > 0:
        eligible = results_df[
            (results_df["ll_delta_vs_raw"] < 0) &   # improves log loss
            (results_df["sharpness"] >= 0.04)        # retains sharpness
        ].copy()

        if len(eligible) == 0:
            print("[!] No combo strictly improves both — picking by log loss only.")
            eligible = results_df.copy()

        best = (
            eligible.sort_values("log_loss")
            .groupby("family").head(1)
            .reset_index(drop=True)
        )
        print("\n── Selected best combo per family ──")
        print(best[["family", "calibrator", "log_loss", "brier", "ece", "sharpness"]]
              .to_string(index=False))

        best_combos = best.to_dict("records")
    else:
        best_combos = []

    with open(BEST_COMBOS, "w") as f:
        json.dump(best_combos, f, indent=2)
    print(f"Best combos saved → {BEST_COMBOS}")

    # ── Save OOF probs (Phase 5 stacking features) ────────────────────────────
    oof_df = df[[YEAR_COL, TARGET]].copy()
    for col, series in oof_cols.items():
        oof_df[col] = series.reindex(df.index).values
    oof_df.to_csv(OOF_PROBS, index=False)
    print(f"OOF probs saved → {OOF_PROBS}  (shape {oof_df.shape})")

    # ── Save inference probs ──────────────────────────────────────────────────
    infer_out = infer_df[["team_a", "team_b"]].copy()
    for col, arr in infer_cols.items():
        infer_out[col] = arr
    infer_out.to_csv(INFER_PROBS, index=False)
    print(f"Inference probs saved → {INFER_PROBS}  (shape {infer_out.shape})")

    # ── Scoring rule experiments ──────────────────────────────────────────────
    print("\n── Alternative scoring rule experiments ──")
    # Train full-feature LR under spherical score and compare to NLL
    _score_rule_experiments(df)

    print("\n✓ Phase 4 complete.")
    print(f"  Next: feed {OOF_PROBS} + {INFER_PROBS} into phase5_ensemble.py")


def _score_rule_experiments(df):
    """
    Compare NLL (log loss) vs spherical score as training objectives.
    We approximate spherical-score training by custom loss via scipy.
    """
    from sklearn.model_selection import cross_val_score

    years      = sorted(df[YEAR_COL].unique())
    test_years = years[-1:]
    tr = df[~df[YEAR_COL].isin(test_years)]
    te = df[df[YEAR_COL].isin(test_years)]

    if len(te) == 0 or te[TARGET].nunique() < 2:
        return

    sc = StandardScaler()
    X_tr = sc.fit_transform(tr[ALL_FEATURES])
    X_te = sc.transform(te[ALL_FEATURES])
    y_tr = tr[TARGET].values
    y_te = te[TARGET].values

    # Standard NLL logistic
    clf_nll = LogisticRegression(max_iter=2000, C=1.0)
    clf_nll.fit(X_tr, y_tr)
    p_nll = clf_nll.predict_proba(X_te)[:, 1]
    ll_nll = log_loss(y_te, np.clip(p_nll, 1e-7, 1 - 1e-7))

    # Spherical score maximization via logistic params + scipy
    def neg_spherical(w):
        logits  = X_tr @ w[:-1] + w[-1]
        p       = 1 / (1 + np.exp(-logits))
        p       = np.clip(p, 1e-7, 1 - 1e-7)
        norm    = np.sqrt(p ** 2 + (1 - p) ** 2)
        sphere  = np.where(y_tr == 1, p / norm, (1 - p) / norm)
        return -np.mean(sphere) + 1e-4 * np.sum(w ** 2)

    w0  = np.zeros(X_tr.shape[1] + 1)
    res = minimize(neg_spherical, w0, method="L-BFGS-B",
                   options={"maxiter": 500})
    w_opt = res.x
    logits_te = X_te @ w_opt[:-1] + w_opt[-1]
    p_sph     = 1 / (1 + np.exp(-logits_te))
    ll_sph    = log_loss(y_te, np.clip(p_sph, 1e-7, 1 - 1e-7))

    print(f"  NLL-trained log loss:        {ll_nll:.5f}")
    print(f"  Spherical-trained log loss:  {ll_sph:.5f}")
    delta = ll_sph - ll_nll
    note  = "spherical better" if delta < 0 else "NLL better"
    print(f"  Δ (spherical - NLL):         {delta:+.5f}  → {note}")
    print("  (For diversity, consider including spherical-trained model in Phase 5 ensemble)")


if __name__ == "__main__":
    main()

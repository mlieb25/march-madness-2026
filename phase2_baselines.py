"""
Phase 2 — Baseline Models & Ground-Truth Benchmarks
=====================================================
Reads:  data/ml_training_data.csv  (output of Phase 1 / etl.py)
Writes: data/phase2_results.csv          — per-season log loss / Brier table
        data/phase2_calibration_*.png    — reliability curves per model
        data/phase2_bar_to_beat.json     — canonical bar-to-beat scores
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH       = "data/ml_training_data.csv"
RESULTS_PATH    = "data/phase2_results.csv"
BAR_PATH        = "data/phase2_bar_to_beat.json"
PLOT_DIR        = "data"

# Small feature set per the procedure doc
SMALL_FEATURES  = ["adjoe_diff", "adjde_diff", "barthag_diff", "adjt_diff"]
ALL_FEATURES    = [
    "adjoe_diff", "adjde_diff", "barthag_diff",
    "sos_diff",   "wab_diff",   "adjt_diff",
    "adjoe_ratio", "adjde_ratio", "barthag_ratio",
    "sos_ratio",   "wab_ratio",   "adjt_ratio",
]
TARGET          = "favorite_win_flag"
YEAR_COL        = "year"


# ── Helpers ───────────────────────────────────────────────────────────────────
def time_aware_split(df, test_year):
    """Train on seasons up to test_year-2, validate on test_year-1, test on test_year."""
    train = df[df[YEAR_COL] <= test_year - 2]
    val   = df[df[YEAR_COL] == test_year - 1]
    test  = df[df[YEAR_COL] == test_year]
    return train, val, test


def safe_log_loss(y_true, y_prob):
    if len(y_true) == 0 or y_true.nunique() < 2:
        return np.nan
    return log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7))


def safe_brier(y_true, y_prob):
    if len(y_true) == 0:
        return np.nan
    return brier_score_loss(y_true, y_prob)


# ── Model 1: Barthag-Only Logistic (Massey-style) ────────────────────────────
def train_barthag_logistic(X_train, y_train):
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X_train[["barthag_diff"]])
    clf    = LogisticRegression(max_iter=1000)
    clf.fit(X_s, y_train)
    return clf, scaler


def predict_barthag_logistic(clf, scaler, X):
    X_s = scaler.transform(X[["barthag_diff"]])
    return clf.predict_proba(X_s)[:, 1]


# ── Model 2: Small-Feature Logistic Regression ───────────────────────────────
def train_small_logistic(X_train, y_train):
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X_train[SMALL_FEATURES])
    clf    = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_s, y_train)
    return clf, scaler


def predict_small_logistic(clf, scaler, X):
    X_s = scaler.transform(X[SMALL_FEATURES])
    return clf.predict_proba(X_s)[:, 1]


# ── Model 3: Full-Feature Logistic Regression ────────────────────────────────
def train_full_logistic(X_train, y_train):
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X_train[ALL_FEATURES])
    clf    = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_s, y_train)
    return clf, scaler


def predict_full_logistic(clf, scaler, X):
    X_s = scaler.transform(X[ALL_FEATURES])
    return clf.predict_proba(X_s)[:, 1]


# ── Calibrators ──────────────────────────────────────────────────────────────
def apply_calibrators(raw_probs, y_val, y_test):
    """
    Train Platt and Isotonic calibrators on val split,
    then apply to test split.
    Returns dict: {calibrator_name: test_probs}
    """
    results = {"none": raw_probs}  # uncalibrated test probs

    for method in ("sigmoid", "isotonic"):
        if len(y_val) < 10 or y_val.nunique() < 2:
            results[method] = raw_probs
            continue
        # Calibrate using val raw probs
        val_probs   = np.clip(raw_probs[: len(y_val)], 1e-7, 1 - 1e-7)
        test_probs_ = raw_probs[len(y_val):]

        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression as LR

        if method == "sigmoid":
            cal = LR(max_iter=1000)
            cal.fit(val_probs.reshape(-1, 1), y_val)
            calibrated = cal.predict_proba(test_probs_.reshape(-1, 1))[:, 1]
        else:  # isotonic
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(val_probs, y_val)
            calibrated = cal.predict(test_probs_)

        results[method] = calibrated

    return results


# ── Reliability / Calibration Plot ───────────────────────────────────────────
def plot_calibration(model_name, y_true, prob_dict, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for label, probs in prob_dict.items():
        if len(probs) != len(y_true):
            continue
        fraction, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
        ax.plot(mean_pred, fraction, marker="o", label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Reliability Curve — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# ── Main Evaluation Loop ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 2 — Baseline Models & Ground-Truth Benchmarks")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET] + ALL_FEATURES)

    years      = sorted(df[YEAR_COL].unique())
    test_years = [y for y in years if y >= min(years) + 2]

    records = []

    # Accumulate all test-year true labels + probs for global calibration plots
    all_true    = {m: [] for m in ("barthag_lr", "small_lr", "full_lr")}
    all_probs   = {m: [] for m in ("barthag_lr", "small_lr", "full_lr")}

    for test_year in test_years:
        train, val, test = time_aware_split(df, test_year)

        if len(train) < 20 or len(test) == 0:
            continue
        if test[TARGET].nunique() < 2:
            continue

        y_train = train[TARGET]
        y_val   = val[TARGET]
        y_test  = test[TARGET]

        # ── Model 1: Barthag-only LR ─────────────────────────────────────────
        clf1, sc1 = train_barthag_logistic(train, y_train)
        prob1_val  = predict_barthag_logistic(clf1, sc1, val)
        prob1_test = predict_barthag_logistic(clf1, sc1, test)

        # ── Model 2: Small-feature LR ─────────────────────────────────────────
        clf2, sc2 = train_small_logistic(train, y_train)
        prob2_val  = predict_small_logistic(clf2, sc2, val)
        prob2_test = predict_small_logistic(clf2, sc2, test)

        # ── Model 3: Full-feature LR ─────────────────────────────────────────
        clf3, sc3 = train_full_logistic(train, y_train)
        prob3_val  = predict_full_logistic(clf3, sc3, val)
        prob3_test = predict_full_logistic(clf3, sc3, test)

        # Combine val + test for calibration (calibrators trained on val, applied to test)
        combined_true = pd.concat([y_val, y_test]).reset_index(drop=True)

        for model_name, prob_val, prob_test in [
            ("barthag_lr", prob1_val, prob1_test),
            ("small_lr",   prob2_val, prob2_test),
            ("full_lr",    prob3_val, prob3_test),
        ]:
            all_true[model_name].extend(y_test.tolist())
            all_probs[model_name].extend(prob_test.tolist())

            ll_raw    = safe_log_loss(y_test, prob_test)
            bs_raw    = safe_brier(y_test, prob_test)

            # Platt (sigmoid)
            if len(y_val) >= 10 and y_val.nunique() >= 2:
                from sklearn.linear_model import LogisticRegression as LR
                cal_s = LR(max_iter=1000)
                cal_s.fit(np.clip(prob_val, 1e-7, 1 - 1e-7).reshape(-1, 1), y_val)
                prob_platt = cal_s.predict_proba(
                    np.clip(prob_test, 1e-7, 1 - 1e-7).reshape(-1, 1))[:, 1]
            else:
                prob_platt = prob_test

            # Isotonic
            if len(y_val) >= 10 and y_val.nunique() >= 2:
                from sklearn.isotonic import IsotonicRegression
                cal_i = IsotonicRegression(out_of_bounds="clip")
                cal_i.fit(np.clip(prob_val, 1e-7, 1 - 1e-7), y_val)
                prob_iso = cal_i.predict(np.clip(prob_test, 1e-7, 1 - 1e-7))
            else:
                prob_iso = prob_test

            records.append({
                "test_year":       test_year,
                "model":           model_name,
                "n_train":         len(train),
                "n_test":          len(test),
                "log_loss_raw":    round(ll_raw, 5),
                "brier_raw":       round(bs_raw, 5),
                "log_loss_platt":  round(safe_log_loss(y_test, prob_platt), 5),
                "brier_platt":     round(safe_brier(y_test, prob_platt), 5),
                "log_loss_iso":    round(safe_log_loss(y_test, prob_iso), 5),
                "brier_iso":       round(safe_brier(y_test, prob_iso), 5),
            })

    results_df = pd.DataFrame(records)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nPer-year results saved → {RESULTS_PATH}")

    # ── Summary table ────────────────────────────────────────────────────────
    summary = (
        results_df
        .groupby("model")[["log_loss_raw", "brier_raw", "log_loss_platt", "log_loss_iso"]]
        .mean()
        .round(5)
    )
    print("\n── Mean scores across all test seasons ──")
    print(summary.to_string())

    # ── Bar to beat ──────────────────────────────────────────────────────────
    bar = summary.to_dict()
    bar["description"] = (
        "Mean log loss / Brier across all held-out test seasons. "
        "Phase 3+ models must beat full_lr log_loss_raw to earn a spot."
    )
    with open(BAR_PATH, "w") as f:
        json.dump(bar, f, indent=2)
    print(f"Bar-to-beat saved → {BAR_PATH}")
    print(f"  ► Target to beat: full_lr log_loss_raw = {summary.loc['full_lr', 'log_loss_raw']:.5f}")

    # ── Calibration plots ────────────────────────────────────────────────────
    for model_name in ("barthag_lr", "small_lr", "full_lr"):
        y_true_all = np.array(all_true[model_name])
        y_prob_all = np.array(all_probs[model_name])

        if len(y_true_all) == 0 or len(np.unique(y_true_all)) < 2:
            continue

        # Also compute platt + isotonic over the whole stack for the plot
        mid  = len(y_true_all) // 2  # rough val/test split for global calibrators
        y_v  = y_true_all[:mid]
        p_v  = y_prob_all[:mid]
        y_te = y_true_all[mid:]
        p_te = y_prob_all[mid:]

        prob_dict = {"raw": y_prob_all}

        if len(y_v) >= 10 and len(np.unique(y_v)) >= 2:
            from sklearn.linear_model import LogisticRegression as LR
            from sklearn.isotonic import IsotonicRegression
            cal_s = LR(max_iter=1000)
            cal_s.fit(np.clip(p_v, 1e-7, 1 - 1e-7).reshape(-1, 1), y_v)
            prob_dict["platt"] = np.concatenate([
                cal_s.predict_proba(np.clip(p_v, 1e-7, 1 - 1e-7).reshape(-1, 1))[:, 1],
                cal_s.predict_proba(np.clip(p_te, 1e-7, 1 - 1e-7).reshape(-1, 1))[:, 1],
            ])
            cal_i = IsotonicRegression(out_of_bounds="clip")
            cal_i.fit(np.clip(p_v, 1e-7, 1 - 1e-7), y_v)
            prob_dict["isotonic"] = np.concatenate([
                cal_i.predict(np.clip(p_v, 1e-7, 1 - 1e-7)),
                cal_i.predict(np.clip(p_te, 1e-7, 1 - 1e-7)),
            ])

        save_path = f"{PLOT_DIR}/phase2_calibration_{model_name}.png"
        plot_calibration(model_name, y_true_all, prob_dict, save_path)
        print(f"Calibration plot saved → {save_path}")

    print("\n✓ Phase 2 complete.")


if __name__ == "__main__":
    main()

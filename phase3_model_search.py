"""
Phase 3 — Systematic Model Search (Trees, GLMs, GPs, Ensembles)
================================================================
Reads:  data/ml_training_data.csv       (Phase 1 output)
        data/phase2_bar_to_beat.json    (Phase 2 bar)
Writes: data/phase3_cv_results.csv      — full Optuna trial table
        data/phase3_top_models.json     — top-3 candidates per family
        data/phase3_summary.png         — log loss comparison chart
"""

import json
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH        = "data/ml_training_data.csv"
BAR_PATH         = "data/phase2_bar_to_beat.json"
CV_RESULTS_PATH  = "data/phase3_cv_results.csv"
TOP_MODELS_PATH  = "data/phase3_top_models.json"
SUMMARY_PLOT     = "data/phase3_summary.png"

ALL_FEATURES = [
    "adjoe_diff", "adjde_diff", "barthag_diff",
    "sos_diff",   "wab_diff",   "adjt_diff",
    "adjoe_ratio", "adjde_ratio", "barthag_ratio",
    "sos_ratio",   "wab_ratio",   "adjt_ratio",
]
GP_FEATURES  = ["adjoe_diff", "adjde_diff", "barthag_diff", "adjt_diff"]  # reduced for GP
TARGET       = "favorite_win_flag"
YEAR_COL     = "year"

N_TRIALS_PER_FAMILY = 40   # raise for longer runs; 40 is quick but meaningful
TOP_K        = 3            # candidates to keep per family


# ── Rolling-years CV ─────────────────────────────────────────────────────────
def rolling_cv_folds(df, min_train_seasons=1):
    """
    Yield (train_df, val_df) pairs where:
      - train = earliest seasons up to val_year - 1
      - val   = single season
    Ensures no same-season leakage.
    """
    years = sorted(df[YEAR_COL].unique())
    for i, val_year in enumerate(years):
        train_years = years[:i]
        if len(train_years) < min_train_seasons:
            continue
        train = df[df[YEAR_COL].isin(train_years)]
        val   = df[df[YEAR_COL] == val_year]
        if len(val) == 0 or val[TARGET].nunique() < 2:
            continue
        yield train, val


def cv_log_loss(df, predict_fn, features):
    """
    Run rolling CV and return mean log loss.
    `predict_fn` takes (X_train, y_train, X_val) → val_probs (np.ndarray).
    """
    losses = []
    for train, val in rolling_cv_folds(df):
        X_tr = train[features]
        y_tr = train[TARGET]
        X_v  = val[features]
        y_v  = val[TARGET]
        try:
            probs = predict_fn(X_tr, y_tr, X_v)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            losses.append(log_loss(y_v, probs))
        except Exception:
            continue
    return float(np.mean(losses)) if losses else np.nan


# ── ECE helper ───────────────────────────────────────────────────────────────
def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.array(y_true)
    y_prob = np.clip(np.array(y_prob), 1e-7, 1 - 1e-7)
    bins   = np.linspace(0, 1, n_bins + 1)
    ece    = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return ece / len(y_true)


# ══════════════════════════════════════════════════════════════════════════════
# Family 1 — Penalized Logistic / Elastic Net
# ══════════════════════════════════════════════════════════════════════════════
def search_elastic_net(df, n_trials):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  [!] optuna not installed. Run: pip install optuna")
        return []

    records = []

    def objective(trial):
        C       = trial.suggest_float("C",       0.001, 10.0,  log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0,  1.0)

        def predict_fn(X_tr, y_tr, X_v):
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_v_s  = sc.transform(X_v)
            clf = LogisticRegression(
                penalty="elasticnet", solver="saga",
                C=C, l1_ratio=l1_ratio, max_iter=2000
            )
            clf.fit(X_tr_s, y_tr)
            return clf.predict_proba(X_v_s)[:, 1]

        return cv_log_loss(df, predict_fn, ALL_FEATURES)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    for t in study.trials:
        records.append({
            "family": "elastic_net",
            "trial":  t.number,
            "params": json.dumps(t.params),
            "cv_log_loss": t.value,
        })

    return records


# ══════════════════════════════════════════════════════════════════════════════
# Family 2 — XGBoost
# ══════════════════════════════════════════════════════════════════════════════
def search_xgboost(df, n_trials):
    try:
        import xgboost as xgb
    except ImportError:
        print("  [!] xgboost not installed. Run: pip install xgboost")
        return []
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return []

    records = []

    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int  ("n_estimators",   50,  500),
            learning_rate     = trial.suggest_float("learning_rate",   0.01, 0.3, log=True),
            max_depth         = trial.suggest_int  ("max_depth",       2,   8),
            subsample         = trial.suggest_float("subsample",       0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree",0.5, 1.0),
            min_child_weight  = trial.suggest_int  ("min_child_weight",1,   10),
            reg_alpha         = trial.suggest_float("reg_alpha",       1e-8, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda",      1e-8, 10.0, log=True),
            use_label_encoder = False,
            eval_metric       = "logloss",
            verbosity         = 0,
        )

        def predict_fn(X_tr, y_tr, X_v):
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_tr, y_tr)
            return clf.predict_proba(X_v)[:, 1]

        return cv_log_loss(df, predict_fn, ALL_FEATURES)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    for t in study.trials:
        records.append({
            "family": "xgboost",
            "trial":  t.number,
            "params": json.dumps(t.params),
            "cv_log_loss": t.value,
        })

    return records


# ══════════════════════════════════════════════════════════════════════════════
# Family 3 — LightGBM
# ══════════════════════════════════════════════════════════════════════════════
def search_lightgbm(df, n_trials):
    try:
        import lightgbm as lgb
    except ImportError:
        print("  [!] lightgbm not installed. Run: pip install lightgbm")
        return []
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return []

    records = []

    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int  ("n_estimators",    50,  500),
            learning_rate     = trial.suggest_float("learning_rate",    0.01, 0.3, log=True),
            max_depth         = trial.suggest_int  ("max_depth",        2,   8),
            num_leaves        = trial.suggest_int  ("num_leaves",       8,   128),
            subsample         = trial.suggest_float("subsample",        0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_samples = trial.suggest_int  ("min_child_samples",5,   50),
            reg_alpha         = trial.suggest_float("reg_alpha",        1e-8, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda",       1e-8, 10.0, log=True),
            verbosity         = -1,
        )

        def predict_fn(X_tr, y_tr, X_v):
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_tr, y_tr)
            return clf.predict_proba(X_v)[:, 1]

        return cv_log_loss(df, predict_fn, ALL_FEATURES)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    for t in study.trials:
        records.append({
            "family": "lightgbm",
            "trial":  t.number,
            "params": json.dumps(t.params),
            "cv_log_loss": t.value,
        })

    return records


# ══════════════════════════════════════════════════════════════════════════════
# Family 4 — Gaussian Process Classifier (reduced feature set)
# ══════════════════════════════════════════════════════════════════════════════
def search_gp(df, n_trials):
    """
    GPC is expensive, so we use a small Optuna search over kernel length-scale
    and a Matern kernel. Reduced feature set to keep runtime manageable.
    """
    try:
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
    except ImportError:
        print("  [!] sklearn GP not available.")
        return []
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return []

    records = []

    def objective(trial):
        length_scale = trial.suggest_float("length_scale", 0.1, 10.0, log=True)
        nu           = trial.suggest_categorical("nu", [0.5, 1.5, 2.5])
        kernel = C(1.0) * Matern(length_scale=length_scale, nu=nu)

        def predict_fn(X_tr, y_tr, X_v):
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_v_s  = sc.transform(X_v)
            # Subsample training for speed (GP is O(n^3))
            if len(X_tr_s) > 400:
                idx = np.random.choice(len(X_tr_s), 400, replace=False)
                X_tr_s = X_tr_s[idx]
                y_tr   = np.array(y_tr)[idx]
            clf = GaussianProcessClassifier(kernel=kernel, max_iter_predict=100)
            clf.fit(X_tr_s, y_tr)
            return clf.predict_proba(X_v_s)[:, 1]

        return cv_log_loss(df, predict_fn, GP_FEATURES)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=min(n_trials, 15), show_progress_bar=False)

    for t in study.trials:
        records.append({
            "family": "gp",
            "trial":  t.number,
            "params": json.dumps(t.params),
            "cv_log_loss": t.value,
        })

    return records


# ── Summary Plot ─────────────────────────────────────────────────────────────
def plot_summary(cv_df, bar_to_beat, save_path):
    families = cv_df["family"].unique()
    colors   = plt.cm.tab10(np.linspace(0, 1, len(families)))

    fig, ax = plt.subplots(figsize=(10, 5))
    for fam, col in zip(families, colors):
        sub = cv_df[cv_df["family"] == fam].sort_values("cv_log_loss")
        ax.scatter(
            range(len(sub)), sub["cv_log_loss"],
            label=fam, color=col, alpha=0.6, s=30
        )
        # mark best
        best = sub.iloc[0]
        ax.scatter(0, best["cv_log_loss"], color=col, s=120, marker="*", zorder=5)

    ax.axhline(bar_to_beat, color="red", linestyle="--", linewidth=1.5,
               label=f"Phase 2 bar ({bar_to_beat:.4f})")
    ax.set_xlabel("Trial (sorted by log loss within family)")
    ax.set_ylabel("CV Log Loss")
    ax.set_title("Phase 3 — Model Search Results")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Summary plot saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 3 — Systematic Model Search")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET] + ALL_FEATURES)

    # Load Phase 2 bar
    bar_to_beat = 0.65  # fallback
    try:
        with open(BAR_PATH) as f:
            bar_data = json.load(f)
        bar_to_beat = bar_data.get("log_loss_raw", {}).get("full_lr", bar_to_beat)
        if isinstance(bar_to_beat, dict):
            bar_to_beat = list(bar_to_beat.values())[0]
        print(f"Phase 2 bar to beat: {bar_to_beat:.5f}")
    except FileNotFoundError:
        print(f"Phase 2 bar file not found — using fallback {bar_to_beat:.5f}")

    all_records = []

    # ── Run all family searches ───────────────────────────────────────────────
    print(f"\n[1/4] Elastic Net (n_trials={N_TRIALS_PER_FAMILY})")
    all_records += search_elastic_net(df, N_TRIALS_PER_FAMILY)

    print(f"[2/4] XGBoost     (n_trials={N_TRIALS_PER_FAMILY})")
    all_records += search_xgboost(df, N_TRIALS_PER_FAMILY)

    print(f"[3/4] LightGBM    (n_trials={N_TRIALS_PER_FAMILY})")
    all_records += search_lightgbm(df, N_TRIALS_PER_FAMILY)

    print(f"[4/4] Gaussian Process (n_trials=15 max — GP is expensive)")
    all_records += search_gp(df, N_TRIALS_PER_FAMILY)

    if not all_records:
        print("No results recorded — check that optuna and model libraries are installed.")
        return

    cv_df = pd.DataFrame(all_records)
    cv_df = cv_df.dropna(subset=["cv_log_loss"])
    cv_df.to_csv(CV_RESULTS_PATH, index=False)
    print(f"\nFull trial table saved → {CV_RESULTS_PATH}")

    # ── Select top-K per family ───────────────────────────────────────────────
    top_models = {}
    print("\n── Top models per family ──")
    for family, grp in cv_df.groupby("family"):
        top = grp.nsmallest(TOP_K, "cv_log_loss").reset_index(drop=True)
        top_models[family] = top.to_dict("records")
        print(f"\n  {family}:")
        for _, row in top.iterrows():
            beats = "✓ beats bar" if row["cv_log_loss"] < bar_to_beat else "✗ above bar"
            print(f"    trial={int(row['trial']):3d}  "
                  f"cv_log_loss={row['cv_log_loss']:.5f}  {beats}")
            print(f"    params={row['params']}")

    with open(TOP_MODELS_PATH, "w") as f:
        json.dump(top_models, f, indent=2)
    print(f"\nTop-{TOP_K} models per family saved → {TOP_MODELS_PATH}")

    # ── Calibration diagnostics on best model per family ─────────────────────
    print("\n── Calibration diagnostics (best per family on last 2 seasons) ──")
    years      = sorted(df[YEAR_COL].unique())
    test_years = years[-2:]
    train_df   = df[~df[YEAR_COL].isin(test_years)]
    test_df    = df[df[YEAR_COL].isin(test_years)]

    if len(test_df) > 0 and test_df[TARGET].nunique() >= 2:
        for family, candidates in top_models.items():
            best_params = json.loads(candidates[0]["params"])
            probs = _refit_and_predict(family, best_params, train_df, test_df)
            if probs is None:
                continue
            ece = expected_calibration_error(test_df[TARGET].values, probs)
            ll  = log_loss(test_df[TARGET].values, np.clip(probs, 1e-7, 1 - 1e-7))
            bs  = brier_score_loss(test_df[TARGET].values, probs)
            print(f"  {family:15s}  log_loss={ll:.5f}  brier={bs:.5f}  ECE={ece:.5f}")

    # ── Summary chart ────────────────────────────────────────────────────────
    plot_summary(cv_df, bar_to_beat, SUMMARY_PLOT)

    print("\n✓ Phase 3 complete.")
    print(f"  Next: feed {TOP_MODELS_PATH} into phase4_calibration.py")


def _refit_and_predict(family, params, train_df, test_df):
    """Refit the best model for a family on train_df, predict on test_df."""
    X_tr = train_df[ALL_FEATURES]
    y_tr = train_df[TARGET]
    X_te = test_df[ALL_FEATURES]

    try:
        if family == "elastic_net":
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)
            clf = LogisticRegression(
                penalty="elasticnet", solver="saga",
                C=params.get("C", 1.0),
                l1_ratio=params.get("l1_ratio", 0.5),
                max_iter=2000
            )
            clf.fit(X_tr_s, y_tr)
            return clf.predict_proba(X_te_s)[:, 1]

        elif family == "xgboost":
            import xgboost as xgb
            clf = xgb.XGBClassifier(
                **{k: v for k, v in params.items()},
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0
            )
            clf.fit(X_tr, y_tr)
            return clf.predict_proba(X_te)[:, 1]

        elif family == "lightgbm":
            import lightgbm as lgb
            clf = lgb.LGBMClassifier(**{k: v for k, v in params.items()}, verbosity=-1)
            clf.fit(X_tr, y_tr)
            return clf.predict_proba(X_te)[:, 1]

        elif family == "gp":
            from sklearn.gaussian_process import GaussianProcessClassifier
            from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
            sc = StandardScaler()
            X_tr_gp = sc.fit_transform(X_tr[GP_FEATURES])
            X_te_gp = sc.transform(X_te[GP_FEATURES])
            if len(X_tr_gp) > 400:
                idx = np.random.choice(len(X_tr_gp), 400, replace=False)
                X_tr_gp = X_tr_gp[idx]
                y_tr_gp = np.array(y_tr)[idx]
            else:
                y_tr_gp = np.array(y_tr)
            kernel = C(1.0) * Matern(
                length_scale=params.get("length_scale", 1.0),
                nu=params.get("nu", 1.5)
            )
            clf = GaussianProcessClassifier(kernel=kernel, max_iter_predict=100)
            clf.fit(X_tr_gp, y_tr_gp)
            return clf.predict_proba(X_te_gp)[:, 1]

    except Exception as e:
        print(f"    [!] refit failed for {family}: {e}")
        return None


if __name__ == "__main__":
    main()

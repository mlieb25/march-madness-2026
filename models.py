"""
Phase 2 & 3 — Baseline (Logistic Regression) and XGBoost model search.
Uses config.TEST_YEAR for holdout; Phase 3 uses time-based CV and writes phase3_top_models.json.
"""
import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.base import clone
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

import config
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load training and inference datasets."""
    try:
        train = pd.read_csv("data/ml_training_data.csv")
        inf = pd.read_csv("data/ml_inference_data_2026.csv")
        return train, inf
    except Exception as e:
        print("Data load error:", e)
        return None, None


def evaluate(y_true, y_pred_prob, model_name="Model"):
    """Calculate and print standard tournament ML metrics."""
    ll  = log_loss(y_true, y_pred_prob)
    bs  = brier_score_loss(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Log Loss:    {ll:.4f}")
    print(f"Brier Score: {bs:.4f}")
    print(f"ROC-AUC:     {auc:.4f}")
    return ll, bs, auc


def time_based_cv_splits(train_df, year_col="year"):
    """
    Yield (train_idx, val_idx) as positional indices where train = all years before val_year,
    val = val_year. Uses only years that have at least MIN_TRAIN_YEARS before them.
    """
    years = sorted(train_df[year_col].unique())
    train_df = train_df.reset_index(drop=True)
    for i in range(config.MIN_TRAIN_YEARS, len(years)):
        val_year = years[i]
        train_years = years[:i]
        train_pos = train_df.index[train_df[year_col].isin(train_years)].tolist()
        val_pos = train_df.index[train_df[year_col] == val_year].tolist()
        if len(val_pos) < 5:
            continue
        yield train_pos, val_pos


def run_phase2_baseline(train, inf, features, target):
    """Phase 2: Baseline Logistic Regression Pipeline."""
    print("\nStarting Phase 2: Baseline Logistic Regression...")
    test_year = config.TEST_YEAR

    train_split = train[train["year"] < test_year]
    test_split  = train[train["year"] == test_year]

    X_train, y_train = train_split[features], train_split[target]
    X_test,  y_test  = test_split[features],  test_split[target]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    y_test_probs = lr.predict_proba(X_test_scaled)[:, 1]
    evaluate(y_test, y_test_probs, model_name="Baseline Logistic Regression")

    print("Generating Baseline predictions for 2026...")
    X_full_scaled = scaler.fit_transform(train[features])
    lr_final = LogisticRegression(max_iter=1000, random_state=42)
    lr_final.fit(X_full_scaled, train[target])

    X_inf = inf[features]
    X_inf_scaled = scaler.transform(X_inf)

    inf_results = inf[["team_a", "team_b"]].copy()
    inf_results["baseline_prob_a_wins"] = lr_final.predict_proba(X_inf_scaled)[:, 1]
    inf_results.to_csv("data/baseline_predictions_2026.csv", index=False)
    print("Saved -> data/baseline_predictions_2026.csv")

    os.makedirs("data/models", exist_ok=True)
    joblib.dump({"scaler": scaler, "model": lr_final}, "data/models/baseline_lr.pkl")
    print("Saved model -> data/models/baseline_lr.pkl")


def run_phase3_xgboost(train, inf, features, target):
    """Phase 3: XGBoost with time-based CV grid search. Writes phase3_top_models.json."""
    print("\nStarting Phase 3: Systematic Model Search (XGBoost)...")
    test_year = config.TEST_YEAR

    train_split = train[train["year"] < test_year]
    test_split  = train[train["year"] == test_year]

    X_train, y_train = train_split[features], train_split[target]
    X_test,  y_test  = test_split[features],  test_split[target]

    # Time-based CV: train on earlier years, validate on next year (multiple folds)
    train_pre_holdout = train[train["year"] < test_year].copy()
    train_pre_holdout = train_pre_holdout.reset_index(drop=True)
    cv_splits = list(time_based_cv_splits(train_pre_holdout))
    if len(cv_splits) < 2:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=min(3, len(train_pre_holdout) // 10 or 2), shuffle=False)
        cv_splits = list(kf.split(train_pre_holdout))
        print("  [!] Fewer than 2 time-based folds; using KFold for hyperparameter search.")

    param_grid = {
        "max_depth": [2, 3, 4],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [50, 100, 200],
        "subsample": [0.8, 1.0],
    }

    xgb = XGBClassifier(eval_metric="logloss", random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="neg_log_loss",
        cv=cv_splits,
        verbose=1,
        n_jobs=-1,
    )

    # GridSearchCV with custom cv expects indices; our splits are indices into train_pre_holdout
    X_cv = train_pre_holdout[features]
    y_cv = train_pre_holdout[target]
    print("Running GridSearchCV (time-based folds)...")
    grid_search.fit(X_cv, y_cv)

    best_xgb = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_score = -grid_search.best_score_
    print(f"Best Params: {best_params}")
    print(f"Best CV log loss: {cv_score:.4f}")

    # Evaluate on holdout year
    y_test_probs = best_xgb.predict_proba(X_test)[:, 1]
    holdout_ll, _, _ = evaluate(y_test, y_test_probs, model_name="Optimized XGBoost")

    # Persist phase3_top_models.json for phase4.py
    os.makedirs("data", exist_ok=True)
    phase3_out = {
        "xgboost": [
            {
                "family": "xgboost",
                "params": json.dumps(best_params),
                "cv_log_loss": cv_score,
                "holdout_log_loss": holdout_ll,
            }
        ]
    }
    with open("data/phase3_top_models.json", "w") as f:
        json.dump(phase3_out, f, indent=2)
    print("Saved -> data/phase3_top_models.json")

    print("Generating XGBoost predictions for 2026...")
    best_xgb_final = clone(best_xgb)
    best_xgb_final.fit(train[features], train[target])

    X_inf = inf[features]
    inf_results = inf[["team_a", "team_b"]].copy()
    inf_results["xgb_prob_a_wins"] = best_xgb_final.predict_proba(X_inf)[:, 1]
    inf_results.to_csv("data/xgb_predictions_2026.csv", index=False)
    print("Saved -> data/xgb_predictions_2026.csv")

    os.makedirs("data/models", exist_ok=True)
    joblib.dump(best_xgb_final, "data/models/xgb_best.pkl")
    print("Saved model -> data/models/xgb_best.pkl")


def run_multi_year_holdout(train, features, target):
    """Optional: report mean/std log loss across multiple holdout years."""
    print("\n--- Multi-year holdout (Phase 2 & 3 style) ---")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    years = sorted(train["year"].unique())
    lr_losses = []
    for test_year in config.HOLDOUT_YEARS:
        if test_year not in years:
            continue
        train_split = train[train["year"] < test_year]
        test_split = train[train["year"] == test_year]
        if len(train_split) < 20 or len(test_split) < 5:
            continue
        X_tr = train_split[features]
        y_tr = train_split[target]
        X_te = test_split[features]
        y_te = test_split[target]
        scaler = StandardScaler()
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(scaler.fit_transform(X_tr), y_tr)
        p = lr.predict_proba(scaler.transform(X_te))[:, 1]
        ll = log_loss(y_te, np.clip(p, 1e-7, 1 - 1e-7))
        lr_losses.append(ll)
        print(f"  LR holdout {test_year}: log loss = {ll:.4f}")
    if lr_losses:
        print(f"  LR multi-year: mean = {np.mean(lr_losses):.4f}, std = {np.std(lr_losses):.4f}")


if __name__ == "__main__":
    train, inf = load_data()
    if train is not None:
        features = [
            "adjoe_diff", "adjde_diff", "barthag_diff", "sos_diff", "wab_diff", "adjt_diff",
            "adjoe_ratio", "adjde_ratio", "barthag_ratio", "sos_ratio", "wab_ratio", "adjt_ratio",
        ]
        target = "favorite_win_flag"

        run_phase2_baseline(train, inf, features, target)
        run_phase3_xgboost(train, inf, features, target)
        run_multi_year_holdout(train, features, target)
        print("\nPhase 2 and 3 scripts complete.")

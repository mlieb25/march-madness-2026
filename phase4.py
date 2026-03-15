"""
Phase 4 (Quick pipeline) — Calibration for the README flow: models.py → phase4.py → phase5.
This script belongs to the Quick pipeline. For the Full pipeline (multi-model, rolling OOF),
use phase4_calibration.py instead.
Writes calibrated_predictions_2026.csv and bridge files for Phase 5: phase4_oof_probs.csv,
phase4_inference_probs.csv, phase4_best_combos.json.
"""
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

import config
import warnings
warnings.filterwarnings("ignore")


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
    ll = log_loss(y_true, y_pred_prob)
    bs = brier_score_loss(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Log Loss:    {ll:.4f}")
    print(f"Brier Score: {bs:.4f}")
    print(f"ROC AUC:     {auc:.4f}")
    return ll, bs, auc


def run_phase4_calibration(train, inf, features, target):
    """Phase 4: Calibration & bridge outputs for Phase 5."""
    print("\nStarting Phase 4: Model Calibration (Quick pipeline)...")
    test_year = config.TEST_YEAR

    train_split = train[train["year"] < test_year]
    test_split  = train[train["year"] == test_year]

    X_train, y_train = train_split[features], train_split[target]
    X_test,  y_test  = test_split[features],  test_split[target]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # --- 1. Base Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
    evaluate(y_test, y_lr_prob, model_name="Uncalibrated Logistic Regression")

    cal_lr = CalibratedClassifierCV(lr, method="isotonic", cv=3)
    cal_lr.fit(X_train_scaled, y_train)
    y_cal_lr = cal_lr.predict_proba(X_test_scaled)[:, 1]
    test_ll, test_bs, _ = evaluate(y_test, y_cal_lr, model_name="Isotonic Calibrated Logistic Regression")

    cal_lr_sig = CalibratedClassifierCV(lr, method="sigmoid", cv=3)
    cal_lr_sig.fit(X_train_scaled, y_train)
    evaluate(y_test, cal_lr_sig.predict_proba(X_test_scaled)[:, 1], model_name="Sigmoid Calibrated Logistic Regression")

    # --- 2. Base XGBoost (params from phase3_top_models.json or fallback) ---
    _FALLBACK_XGB = {"learning_rate": 0.01, "max_depth": 4, "n_estimators": 50, "subsample": 0.8}
    try:
        with open("data/phase3_top_models.json") as _f:
            _top = json.load(_f)
        _best_params = json.loads(_top.get("xgboost", [{}])[0].get("params", "{}"))
        if not _best_params:
            raise ValueError("empty params")
    except Exception as _e:
        print(f"[!] Could not load phase3_top_models.json XGBoost params ({_e}); using fallback.")
        _best_params = _FALLBACK_XGB
    print(f"    XGBoost params: {_best_params}")
    xgb = XGBClassifier(**_best_params, eval_metric="logloss", random_state=42)
    xgb.fit(X_train, y_train)
    y_xgb_prob = xgb.predict_proba(X_test)[:, 1]
    evaluate(y_test, y_xgb_prob, model_name="Uncalibrated XGBoost")

    cal_xgb = CalibratedClassifierCV(xgb, method="isotonic", cv=3)
    cal_xgb.fit(X_train, y_train)
    evaluate(y_test, cal_xgb.predict_proba(X_test)[:, 1], model_name="Isotonic Calibrated XGBoost")

    cal_xgb_sig = CalibratedClassifierCV(xgb, method="sigmoid", cv=3)
    cal_xgb_sig.fit(X_train, y_train)
    evaluate(y_test, cal_xgb_sig.predict_proba(X_test)[:, 1], model_name="Sigmoid Calibrated XGBoost")

    # --- 3. Final 2026 predictions (Isotonic LR) ---
    print("\nGenerating final calibrated predictions for 2026 (Isotonic LR)...")
    X_full_scaled = scaler.fit_transform(train[features])
    lr_final = LogisticRegression(max_iter=1000, random_state=42)
    cal_lr_final = CalibratedClassifierCV(lr_final, method="isotonic", cv=3)
    cal_lr_final.fit(X_full_scaled, train[target])

    X_inf = inf[features]
    X_inf_scaled = scaler.transform(X_inf)

    inf_results = inf[["team_a", "team_b"]].copy()
    inf_results["calibrated_prob_a_wins"] = cal_lr_final.predict_proba(X_inf_scaled)[:, 1]
    inf_results.to_csv("data/calibrated_predictions_2026.csv", index=False)
    print("Saved -> data/calibrated_predictions_2026.csv")

    # --- 4. Bridge for Phase 5: OOF probs, inference probs, best_combos ---
    # OOF: 3-fold CV on train (year < test_year) for logistic_isotonic; test year gets holdout preds
    oof_probs = np.full(len(train), np.nan)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(train_split):
        X_tr = scaler.fit_transform(train_split[features].iloc[tr_idx])
        y_tr = train_split[target].iloc[tr_idx]
        X_val = scaler.transform(train_split[features].iloc[val_idx])
        lr_f = LogisticRegression(max_iter=1000, random_state=42)
        cal_f = CalibratedClassifierCV(lr_f, method="isotonic", cv=2)
        cal_f.fit(X_tr, y_tr)
        orig_index = train_split.index[val_idx]
        pos_in_train = train.index.get_indexer(orig_index)
        for i, p in enumerate(pos_in_train):
            oof_probs[p] = cal_f.predict_proba(X_val)[:, 1][i]

    # Fill test year with holdout predictions (so Phase 5 can evaluate ensemble on holdout)
    for i, idx in enumerate(test_split.index):
        pos = train.index.get_loc(idx)
        oof_probs[pos] = y_cal_lr[i]

    oof_df = train[["year", target]].copy()
    oof_df["logistic_isotonic"] = oof_probs
    oof_df.to_csv("data/phase4_oof_probs.csv", index=False)
    print("Saved -> data/phase4_oof_probs.csv (bridge for Phase 5)")

    infer_probs = inf[["team_a", "team_b"]].copy()
    infer_probs["logistic_isotonic"] = cal_lr_final.predict_proba(X_inf_scaled)[:, 1]
    infer_probs.to_csv("data/phase4_inference_probs.csv", index=False)
    print("Saved -> data/phase4_inference_probs.csv (bridge for Phase 5)")

    best_combos = [
        {"family": "logistic", "calibrator": "isotonic", "log_loss": float(test_ll), "brier": float(test_bs)}
    ]
    with open("data/phase4_best_combos.json", "w") as f:
        json.dump(best_combos, f, indent=2)
    print("Saved -> data/phase4_best_combos.json (bridge for Phase 5)")


if __name__ == "__main__":
    train, inf = load_data()
    if train is not None:
        features = [
            "adjoe_diff", "adjde_diff", "barthag_diff", "sos_diff", "wab_diff", "adjt_diff",
            "adjoe_ratio", "adjde_ratio", "barthag_ratio", "sos_ratio", "wab_ratio", "adjt_ratio",
        ]
        target = "favorite_win_flag"
        run_phase4_calibration(train, inf, features, target)
        print("\nPhase 4 script complete.")

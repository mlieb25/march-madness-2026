import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
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
    ll = log_loss(y_true, y_pred_prob)
    bs = brier_score_loss(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Log Loss:    {ll:.4f}")
    print(f"Brier Score: {bs:.4f}")
    print(f"ROC AUC:     {auc:.4f}")
    return ll, bs, auc

def run_phase4_calibration(train, inf, features, target):
    """Phase 4: Calibration & Probabilistic Sharpness."""
    print("\nStarting Phase 4: Model Calibration...")
    
    # Time-aware split: Train on 2011-2013, Test on 2014
    train_split = train[train['year'] < 2014]
    test_split  = train[train['year'] == 2014]
    
    X_train, y_train = train_split[features], train_split[target]
    X_test,  y_test  = test_split[features],  test_split[target]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    # --- 1. Base Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
    evaluate(y_test, y_lr_prob, model_name="Uncalibrated Logistic Regression")
    
    # Calibrate LR (Isotonic)
    cal_lr = CalibratedClassifierCV(lr, method='isotonic', cv=3)
    cal_lr.fit(X_train_scaled, y_train)
    evaluate(y_test, cal_lr.predict_proba(X_test_scaled)[:, 1], model_name="Isotonic Calibrated Logistic Regression")
    
    # Calibrate LR (Platt/Sigmoid)
    cal_lr_sig = CalibratedClassifierCV(lr, method='sigmoid', cv=3)
    cal_lr_sig.fit(X_train_scaled, y_train)
    evaluate(y_test, cal_lr_sig.predict_proba(X_test_scaled)[:, 1], model_name="Sigmoid Calibrated Logistic Regression")
    
    # --- 2. Base XGBoost ---
    # Load best hyperparameters from Phase 3 grid-search output
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
    xgb = XGBClassifier(**_best_params, eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    y_xgb_prob = xgb.predict_proba(X_test)[:, 1]
    evaluate(y_test, y_xgb_prob, model_name="Uncalibrated XGBoost")
    
    # Calibrate XGBoost (Isotonic)
    cal_xgb = CalibratedClassifierCV(xgb, method='isotonic', cv=3)
    cal_xgb.fit(X_train, y_train)
    evaluate(y_test, cal_xgb.predict_proba(X_test)[:, 1], model_name="Isotonic Calibrated XGBoost")
    
    # Calibrate XGBoost (Platt/Sigmoid)
    cal_xgb_sig = CalibratedClassifierCV(xgb, method='sigmoid', cv=3)
    cal_xgb_sig.fit(X_train, y_train)
    evaluate(y_test, cal_xgb_sig.predict_proba(X_test)[:, 1], model_name="Sigmoid Calibrated XGBoost")
    
    # --- 3. Generate Final 2026 Predictions ---
    # We will output using the Isotonic Calibrated Logistic Regression, 
    # as strict linear models typically calibrate cleanly and don't overfit
    print("\nGenerating final calibrated predictions for 2026 (Isotonic LR)...")
    
    X_full_scaled = scaler.fit_transform(train[features])
    lr_final = LogisticRegression(max_iter=1000, random_state=42)
    cal_lr_final = CalibratedClassifierCV(lr_final, method='isotonic', cv=3)
    cal_lr_final.fit(X_full_scaled, train[target])
    
    X_inf = inf[features]
    X_inf_scaled = scaler.transform(X_inf)
    
    inf_results = inf[['team_a', 'team_b']].copy()
    inf_results['calibrated_prob_a_wins'] = cal_lr_final.predict_proba(X_inf_scaled)[:, 1]
    inf_results.to_csv("data/calibrated_predictions_2026.csv", index=False)
    print("Saved -> data/calibrated_predictions_2026.csv")


if __name__ == "__main__":
    train, inf = load_data()
    if train is not None:
        features = [
            'adjoe_diff', 'adjde_diff', 'barthag_diff', 'sos_diff', 'wab_diff', 'adjt_diff',
            'adjoe_ratio', 'adjde_ratio', 'barthag_ratio', 'sos_ratio', 'wab_ratio', 'adjt_ratio'
        ]
        target = 'favorite_win_flag'
        
        run_phase4_calibration(train, inf, features, target)
        print("\nPhase 4 script complete.")

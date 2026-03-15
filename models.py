import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.base import clone
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
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

def run_phase2_baseline(train, inf, features, target):
    """Phase 2: Baseline Logistic Regression Pipeline."""
    print("\nStarting Phase 2: Baseline Logistic Regression...")
    
    # Time-aware split: Train on 2011-2013, Test on 2014
    train_split = train[train['year'] < 2014]
    test_split  = train[train['year'] == 2014]
    
    X_train, y_train = train_split[features], train_split[target]
    X_test,  y_test  = test_split[features],  test_split[target]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    # Train Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_test_probs = lr.predict_proba(X_test_scaled)[:, 1]
    evaluate(y_test, y_test_probs, model_name="Baseline Logistic Regression")
    
    # Generate 2026 Predictions
    print("Generating Baseline predictions for 2026...")
    # Retrain on full historical data for final inference
    X_full_scaled = scaler.fit_transform(train[features])
    lr_final = LogisticRegression(max_iter=1000, random_state=42)
    lr_final.fit(X_full_scaled, train[target])
    
    X_inf = inf[features]
    X_inf_scaled = scaler.transform(X_inf)
    
    inf_results = inf[['team_a', 'team_b']].copy()
    inf_results['baseline_prob_a_wins'] = lr_final.predict_proba(X_inf_scaled)[:, 1]
    inf_results.to_csv("data/baseline_predictions_2026.csv", index=False)
    print("Saved -> data/baseline_predictions_2026.csv")

    # Persist model + scaler for downstream phases
    os.makedirs("data/models", exist_ok=True)
    joblib.dump({"scaler": scaler, "model": lr_final}, "data/models/baseline_lr.pkl")
    print("Saved model -> data/models/baseline_lr.pkl")

def run_phase3_xgboost(train, inf, features, target):
    """Phase 3: Systematic Model Search via XGBoost."""
    print("\nStarting Phase 3: Systematic Model Search (XGBoost)...")
    
    train_split = train[train['year'] < 2014]
    test_split  = train[train['year'] == 2014]
    
    X_train, y_train = train_split[features], train_split[target]
    X_test,  y_test  = test_split[features],  test_split[target]
    
    # Define generic hyperparameter grid for optimization
    param_grid = {
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0]
    }
    
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    
    grid_search = GridSearchCV(
        estimator=xgb, 
        param_grid=param_grid, 
        scoring='neg_log_loss', 
        cv=3, 
        verbose=1,
        n_jobs=-1
    )
    
    print("Running GridSearchCV...")
    grid_search.fit(X_train, y_train)
    
    best_xgb = grid_search.best_estimator_
    print(f"Best Params: {grid_search.best_params_}")
    
    # Evaluate best model
    y_test_probs = best_xgb.predict_proba(X_test)[:, 1]
    evaluate(y_test, y_test_probs, model_name="Optimized XGBoost")
    
    # Generate 2026 Predictions — clone best estimator to avoid duplicate-kwarg TypeError
    print("Generating XGBoost predictions for 2026...")
    best_xgb_final = clone(best_xgb)
    best_xgb_final.fit(train[features], train[target])
    
    X_inf = inf[features]
    inf_results = inf[['team_a', 'team_b']].copy()
    inf_results['xgb_prob_a_wins'] = best_xgb_final.predict_proba(X_inf)[:, 1]
    inf_results.to_csv("data/xgb_predictions_2026.csv", index=False)
    print("Saved -> data/xgb_predictions_2026.csv")

    # Persist model for downstream phases
    os.makedirs("data/models", exist_ok=True)
    joblib.dump(best_xgb_final, "data/models/xgb_best.pkl")
    print("Saved model -> data/models/xgb_best.pkl")

if __name__ == "__main__":
    train, inf = load_data()
    if train is not None:
        # Define the 12 features from Phase 1
        features = [
            'adjoe_diff', 'adjde_diff', 'barthag_diff', 'sos_diff', 'wab_diff', 'adjt_diff',
            'adjoe_ratio', 'adjde_ratio', 'barthag_ratio', 'sos_ratio', 'wab_ratio', 'adjt_ratio'
        ]
        target = 'favorite_win_flag'
        
        run_phase2_baseline(train, inf, features, target)
        run_phase3_xgboost(train, inf, features, target)
        print("\nPhase 2 and 3 scripts complete.")

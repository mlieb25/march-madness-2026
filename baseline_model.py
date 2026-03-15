#!/usr/bin/env python3
"""
March Madness Baseline Model

Simple seed-based and logistic regression models to establish baseline performance.

Author: Mitchell Liebrecht  
Date: March 15, 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MARCH MADNESS BASELINE MODEL")
print("=" * 80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_DIR = Path('processed_data')
OUTPUT_DIR = Path('models')
OUTPUT_DIR.mkdir(exist_ok=True)

TEST_SEASONS = [2022, 2023, 2024, 2025]  # Hold out recent seasons for validation

# ============================================================================
# 1. LOAD TOURNAMENT GAMES DATA
# ============================================================================

print("[1/5] Loading tournament games...")

try:
    tourney_games = pd.read_csv(PROCESSED_DIR / 'tournament_games_features.csv')
    print(f"  ✓ Loaded {len(tourney_games)} tournament games")
except FileNotFoundError:
    print("  ✗ Error: tournament_games_features.csv not found!")
    print("  Please run build_master_dataset.py first.")
    exit(1)

print()

# ============================================================================
# 2. PREPARE DATA FOR MODELING
# ============================================================================

print("[2/5] Preparing data for modeling...")

# Create label (1 = Team1 won, 0 = Team2 won)
# We'll create a balanced dataset by including both perspectives

# Perspective 1: Winner vs Loser (label = 1)
df1 = tourney_games[[
    'Season', 'WTeamID', 'LTeamID',
    'SeedNum_W', 'SeedNum_L', 'SeedDiff',
    'WinPct_W', 'WinPct_L', 'WinPctDiff',
    'PointDiff_W', 'PointDiff_L', 'PointDiffDiff'
]].copy()
df1.columns = [
    'Season', 'Team1ID', 'Team2ID',
    'Team1_Seed', 'Team2_Seed', 'Seed_Diff',
    'Team1_WinPct', 'Team2_WinPct', 'WinPct_Diff',
    'Team1_PointDiff', 'Team2_PointDiff', 'PointDiff_Diff'
]
df1['Label'] = 1

# Perspective 2: Loser vs Winner (label = 0)  
df2 = tourney_games[[
    'Season', 'LTeamID', 'WTeamID',
    'SeedNum_L', 'SeedNum_W', 'SeedDiff',
    'WinPct_L', 'WinPct_W', 'WinPctDiff',
    'PointDiff_L', 'PointDiff_W', 'PointDiffDiff'
]].copy()
df2.columns = [
    'Season', 'Team1ID', 'Team2ID',
    'Team1_Seed', 'Team2_Seed', 'Seed_Diff',
    'Team1_WinPct', 'Team2_WinPct', 'WinPct_Diff',
    'Team1_PointDiff', 'Team2_PointDiff', 'PointDiff_Diff'
]
# Flip differentials
df2['Seed_Diff'] = -df2['Seed_Diff']
df2['WinPct_Diff'] = -df2['WinPct_Diff']
df2['PointDiff_Diff'] = -df2['PointDiff_Diff']
df2['Label'] = 0

# Combine both perspectives
modeling_df = pd.concat([df1, df2], ignore_index=True)

# Add advanced features if available
if 'eFG_W' in tourney_games.columns:
    print("  ✓ Adding Four Factors features...")
    
    # Add for perspective 1
    df1_adv = tourney_games[['eFGDiff', 'TOV_RateDiff']].copy()
    df1_adv.columns = ['eFG_Diff', 'TOV_Diff']
    
    # Add for perspective 2 (flipped)
    df2_adv = tourney_games[['eFGDiff', 'TOV_RateDiff']].copy()
    df2_adv.columns = ['eFG_Diff', 'TOV_Diff']
    df2_adv = -df2_adv
    
    # Combine
    adv_features = pd.concat([df1_adv, df2_adv], ignore_index=True)
    modeling_df = pd.concat([modeling_df, adv_features], axis=1)

print(f"  ✓ Created balanced dataset: {len(modeling_df)} samples")
print(f"  ✓ Features: {[c for c in modeling_df.columns if c not in ['Season', 'Team1ID', 'Team2ID', 'Label']]}")
print()

# ============================================================================
# 3. SPLIT DATA (TIME-BASED)
# ============================================================================

print("[3/5] Splitting data (time-based cross-validation)...")

# Training  all games before test seasons
train_df = modeling_df[~modeling_df['Season'].isin(TEST_SEASONS)].copy()
test_df = modeling_df[modeling_df['Season'].isin(TEST_SEASONS)].copy()

# Remove rows with missing seeds (teams that didn't make tournament in that year)
train_df = train_df.dropna(subset=['Team1_Seed', 'Team2_Seed'])
test_df = test_df.dropna(subset=['Team1_Seed', 'Team2_Seed'])

print(f"  Train: {len(train_df)} games ({train_df['Season'].min()}-{train_df['Season'].max()})")
print(f"  Test:  {len(test_df)} games ({test_df['Season'].min()}-{test_df['Season'].max()})")
print()

# ============================================================================
# 4. BASELINE MODEL 1: SEED-BASED
# ============================================================================

print("[4/5] Building baseline models...")
print()
print("Model 1: Pure Seed-Based Prediction")
print("-" * 50)

# Historical win rates by seed matchup
seed_matchups = train_df.groupby(['Team1_Seed', 'Team2_Seed'])['Label'].agg(['sum', 'count']).reset_index()
seed_matchups['WinRate'] = seed_matchups['sum'] / seed_matchups['count']
seed_matchups = seed_matchups.rename(columns={'sum': 'Wins', 'count': 'Games'})

print("Historical seed matchup win rates (top 10):")
print(seed_matchups.nlargest(10, 'Games')[['Team1_Seed', 'Team2_Seed', 'WinRate', 'Games']])
print()

# Apply to test set
test_df = test_df.merge(
    seed_matchups[['Team1_Seed', 'Team2_Seed', 'WinRate']], 
    on=['Team1_Seed', 'Team2_Seed'], 
    how='left'
)

# For unseen matchups, use seed differential heuristic
test_df['WinRate'] = test_df['WinRate'].fillna(0.5)

# Adjust based on seed differential for missing matchups
missing_mask = test_df['WinRate'] == 0.5
test_df.loc[missing_mask, 'WinRate'] = 0.5 - (test_df.loc[missing_mask, 'Seed_Diff'] * 0.03)
test_df['WinRate'] = test_df['WinRate'].clip(0.01, 0.99)

# Evaluate
seed_log_loss = log_loss(test_df['Label'], test_df['WinRate'])
seed_accuracy = accuracy_score(test_df['Label'], (test_df['WinRate'] > 0.5).astype(int))
seed_auc = roc_auc_score(test_df['Label'], test_df['WinRate'])

print(f"Seed-Based Model Performance:")
print(f"  Log Loss:  {seed_log_loss:.4f}")
print(f"  Accuracy:  {seed_accuracy:.4f}")
print(f"  AUC:       {seed_auc:.4f}")
print()

# ============================================================================
# 5. BASELINE MODEL 2: LOGISTIC REGRESSION
# ============================================================================

print("Model 2: Logistic Regression")
print("-" * 50)

# Select features for model
feature_cols = ['Seed_Diff', 'WinPct_Diff', 'PointDiff_Diff']

if 'eFG_Diff' in train_df.columns:
    feature_cols.extend(['eFG_Diff', 'TOV_Diff'])

X_train = train_df[feature_cols].fillna(0)
y_train = train_df['Label']

X_test = test_df[feature_cols].fillna(0)
y_test = test_df['Label']

print(f"Features: {feature_cols}")
print(f"Training samples: {len(X_train)}")
print()

# Train logistic regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)

# Predictions
y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

# Evaluate
lr_log_loss = log_loss(y_test, y_pred_proba)
lr_accuracy = accuracy_score(y_test, y_pred)
lr_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Logistic Regression Performance:")
print(f"  Log Loss:  {lr_log_loss:.4f}")
print(f"  Accuracy:  {lr_accuracy:.4f}")
print(f"  AUC:       {lr_auc:.4f}")
print()

print("Feature Importances (coefficients):")
for feat, coef in zip(feature_cols, lr_model.coef_[0]):
    print(f"  {feat:20s}: {coef:+.4f}")
print()

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================

print("Saving results...")

# Save model performance comparison
results = pd.DataFrame({
    'Model': ['Seed-Based', 'Logistic Regression'],
    'Log_Loss': [seed_log_loss, lr_log_loss],
    'Accuracy': [seed_accuracy, lr_accuracy],
    'AUC': [seed_auc, lr_auc]
})
results.to_csv(OUTPUT_DIR / 'baseline_results.csv', index=False)
print(f"  ✓ Saved: {OUTPUT_DIR / 'baseline_results.csv'}")

# Save seed matchup lookup table
seed_matchups.to_csv(OUTPUT_DIR / 'seed_matchup_rates.csv', index=False)
print(f"  ✓ Saved: {OUTPUT_DIR / 'seed_matchup_rates.csv'}")

# Save model
import joblib
joblib.dump(lr_model, OUTPUT_DIR / 'logistic_regression_baseline.pkl')
print(f"  ✓ Saved: {OUTPUT_DIR / 'logistic_regression_baseline.pkl'}")
print()

print("=" * 80)
print("BASELINE MODELS COMPLETE!")
print("=" * 80)
print()
print("Summary:")
print(results.to_string(index=False))
print()
print("Next steps:")
print("  1. Try advanced models (XGBoost, neural networks)")
print("  2. Add more features (external ratings, momentum)")
print("  3. Tune hyperparameters")
print("  4. Ensemble multiple models")
print()

#!/usr/bin/env python3
"""
March Madness Prediction Feature Builder

This script builds pairwise matchup features for making tournament predictions.
It creates features for all possible team matchups in a given season.

Author: Mitchell Liebrecht
Date: March 15, 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MARCH MADNESS PREDICTION FEATURE BUILDER")
print("=" * 80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path('data/march-machine-learning-mania-2026')
PROCESSED_DIR = Path('processed_data')
OUTPUT_DIR = Path('predictions')
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_SEASON = 2026  # Season to make predictions for

print(f"Data directory: {DATA_DIR}")
print(f"Processed data directory: {PROCESSED_DIR}")
print(f"Target season: {TARGET_SEASON}")
print()

# ============================================================================
# 1. LOAD PROCESSED DATA
# ============================================================================

print("[1/4] Loading processed datasets...")

try:
    team_season_stats = pd.read_csv(PROCESSED_DIR / 'team_season_stats.csv')
    print(f"  ✓ Loaded team_season_stats.csv: {team_season_stats.shape}")
except FileNotFoundError:
    print("  ✗ Error: team_season_stats.csv not found!")
    print("  Please run build_master_dataset.py first.")
    exit(1)

teams = pd.read_csv(DATA_DIR / 'MTeams.csv')
print(f"  ✓ Loaded teams: {len(teams)} teams")
print()

# ============================================================================
# 2. GET ACTIVE TEAMS FOR TARGET SEASON
# ============================================================================

print(f"[2/4] Identifying active teams for {TARGET_SEASON}...")

# Get teams active in target season
active_teams = teams[teams['LastD1Season'] >= TARGET_SEASON]['TeamID'].unique()
print(f"  ✓ Found {len(active_teams)} active teams")

# Get their stats from target season (or most recent)
target_stats = team_season_stats[
    (team_season_stats['Season'] == TARGET_SEASON) & 
    (team_season_stats['TeamID'].isin(active_teams))
].copy()

if len(target_stats) == 0:
    print(f"  ! No stats found for {TARGET_SEASON}, using {TARGET_SEASON - 1}")
    target_stats = team_season_stats[
        (team_season_stats['Season'] == TARGET_SEASON - 1) & 
        (team_season_stats['TeamID'].isin(active_teams))
    ].copy()
    target_stats['Season'] = TARGET_SEASON

print(f"  ✓ Loaded stats for {len(target_stats)} teams in {TARGET_SEASON}")
print()

# ============================================================================
# 3. CREATE ALL PAIRWISE MATCHUPS
# ============================================================================

print("[3/4] Creating pairwise matchup features...")

# Get all possible pairwise combinations
team_ids = sorted(target_stats['TeamID'].unique())
pairwise_matchups = list(combinations(team_ids, 2))

print(f"  ✓ Created {len(pairwise_matchups)} possible matchups")

def create_matchup_features(team1_id, team2_id, stats_df):
    """
    Create features for a matchup between team1 and team2.
    Team1 is always the lower ID (as per Kaggle submission format).
    Returns features dict with Team1 - Team2 differentials.
    """
    
    team1_stats = stats_df[stats_df['TeamID'] == team1_id].iloc[0]
    team2_stats = stats_df[stats_df['TeamID'] == team2_id].iloc[0]
    
    features = {
        'Season': team1_stats['Season'],
        'Team1ID': team1_id,
        'Team2ID': team2_id,
        'ID': f"{int(team1_stats['Season'])}_{team1_id}_{team2_id}",
    }
    
    # Basic differential features
    basic_features = ['WinPct', 'PPG', 'OppPPG', 'PointDiff', 'HomeWinPct', 'AwayWinPct']
    for feat in basic_features:
        if feat in team1_stats and not pd.isna(team1_stats[feat]):
            features[f'{feat}_Diff'] = team1_stats[feat] - team2_stats[feat]
    
    # Seed differential (lower seed number = better team)
    if not pd.isna(team1_stats.get('SeedNum', np.nan)):
        features['Seed_Diff'] = team1_stats['SeedNum'] - team2_stats['SeedNum']
        features['Team1_Seed'] = team1_stats['SeedNum']
        features['Team2_Seed'] = team2_stats['SeedNum']
    
    # Advanced stats differential (Four Factors)
    advanced_features = ['eFG', 'FTA_Rate', 'TOV_Rate', 'ORB_Rate', 'TRB', 'Ast', 'Stl', 'Blk']
    for feat in advanced_features:
        if feat in team1_stats and not pd.isna(team1_stats[feat]):
            features[f'{feat}_Diff'] = team1_stats[feat] - team2_stats[feat]
    
    # External ratings differential
    rating_systems = ['POM_Rank', 'SAG_Rank', 'RPI_Rank', 'DOK_Rank', 'COL_Rank']
    for rating in rating_systems:
        if rating in team1_stats and not pd.isna(team1_stats[rating]):
            # Lower rank = better, so flip the sign
            features[f'{rating}_Diff'] = team2_stats[rating] - team1_stats[rating]
    
    # Conference matchup indicator
    if 'ConfAbbrev' in team1_stats:
        features['SameConference'] = int(team1_stats['ConfAbbrev'] == team2_stats['ConfAbbrev'])
    
    return features

# Build features for all matchups
matchup_features = []
for i, (team1, team2) in enumerate(pairwise_matchups):
    if (i + 1) % 10000 == 0:
        print(f"  Processing matchup {i+1}/{len(pairwise_matchups)}...")
    
    features = create_matchup_features(team1, team2, target_stats)
    matchup_features.append(features)

prediction_df = pd.DataFrame(matchup_features)

print(f"  ✓ Created {len(prediction_df)} matchup feature sets")
print(f"  ✓ Features per matchup: {len(prediction_df.columns)}")
print()

# ============================================================================
# 4. CREATE SUBMISSION TEMPLATE
# ============================================================================

print("[4/4] Creating submission template...")

# Create submission format
submission = prediction_df[['ID']].copy()
submission['Pred'] = 0.5  # Baseline prediction (toss-up)

print(f"  ✓ Created submission template with {len(submission)} predictions")
print()

# ============================================================================
# 5. SAVE OUTPUT FILES
# ============================================================================

print("Saving files...")

# Save prediction features
prediction_df.to_csv(OUTPUT_DIR / f'prediction_features_{TARGET_SEASON}.csv', index=False)
print(f"  ✓ Saved: {OUTPUT_DIR / f'prediction_features_{TARGET_SEASON}.csv'}")
print(f"    Shape: {prediction_df.shape}")

# Save submission template
submission.to_csv(OUTPUT_DIR / f'submission_template_{TARGET_SEASON}.csv', index=False)
print(f"  ✓ Saved: {OUTPUT_DIR / f'submission_template_{TARGET_SEASON}.csv'}")
print(f"    Shape: {submission.shape}")

# Save feature list
with open(OUTPUT_DIR / 'feature_list.txt', 'w') as f:
    f.write("PREDICTION FEATURES\n")
    f.write("=" * 50 + "\n\n")
    for i, col in enumerate(prediction_df.columns, 1):
        f.write(f"{i:3d}. {col}\n")

print(f"  ✓ Saved: {OUTPUT_DIR / 'feature_list.txt'}")
print()

print("=" * 80)
print("PREDICTION FEATURES READY!")
print("=" * 80)
print()
print(f"Available files in {OUTPUT_DIR}/:")
print(f"  1. prediction_features_{TARGET_SEASON}.csv - Full feature set for modeling")
print(f"  2. submission_template_{TARGET_SEASON}.csv - Kaggle submission format")
print(f"  3. feature_list.txt - Complete list of features")
print()
print("Sample features:")
for col in prediction_df.columns[:10]:
    print(f"  • {col}")
print(f"  ... and {len(prediction_df.columns) - 10} more")
print()
print("Next steps:")
print("  1. Train model using tournament_games_features.csv")
print(f"  2. Use trained model to predict on prediction_features_{TARGET_SEASON}.csv")
print(f"  3. Fill in Pred column in submission_template_{TARGET_SEASON}.csv")
print("  4. Submit to Kaggle!")
print()

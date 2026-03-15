#!/usr/bin/env python3
"""
ETL v2 - Enhanced Data Pipeline with Master Dataset Integration
================================================================
This version leverages the comprehensive master dataset built by build_master_dataset.py
and combines it with external sources (538, Torvik, NCAA NET) for maximum feature richness.

Inputs:
    - processed_data/team_season_stats.csv     (from build_master_dataset.py)
    - processed_data/tournament_games_features.csv
    - data/fivethirtyeight_forecasts.csv      (external)
    - data/barttorvik_historical.csv          (external)
    - data/ncaa_net.csv                       (external)

Outputs:
    - data/ml_training_data_v2.csv            (enhanced training set)
    - data/ml_inference_data_2026_v2.csv      (enhanced 2026 predictions)
    - data/etl_v2_summary.json                (data quality metrics)

Author: Mitchell Liebrecht
Date: March 15, 2026
"""

import pandas as pd
import numpy as np
import itertools
import json
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ETL V2 - MASTER DATASET INTEGRATION")
print("=" * 80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_DIR = Path('processed_data')
DATA_DIR = Path('data')
KAGGLE_DIR = DATA_DIR / 'march-machine-learning-mania-2026'

# ============================================================================
# NAME NORMALIZATION (Enhanced from original ETL)
# ============================================================================

def normalize_name(name):
    """Normalize team names for cross-source joining."""
    if pd.isna(name):
        return ""
    name = str(name).lower().strip()

    # Pass 1: punctuation cleanup
    name = name.replace("'", "").replace("-", " ").replace("&", "and")
    name = name.replace("(", "").replace(")", "")
    name = re.sub(r"\bst\.", "st", name)
    name = name.replace(" state", " st")
    name = re.sub(r"\buniversity$", "", name).strip()
    name = re.sub(r"\buniv$", "", name).strip()

    # Pass 2: specific mappings
    exact_map = [
        (r"\bnorth carolina st\b", "nc st"),
        (r"\bnorth carolina\b", "unc"),
        (r"\blouisiana st\b", "lsu"),
        (r"\bconnecticut\b", "uconn"),
        (r"\bsouthern california\b", "usc"),
        (r"\bcentral florida\b", "ucf"),
        (r"\bsouthern methodist\b", "smu"),
        (r"\btexas christian\b", "tcu"),
        (r"\bmassachusetts\b", "umass"),
        (r"\bpennsylvania\b", "penn"),
        (r"\bbrigham young\b", "byu"),
        (r"\bvirginia commonwealth\b", "vcu"),
        (r"\bstephen f austin\b", "stephen f austin"),
        (r"\bsaint marys  ca\b", "saint marys"),
        (r"\bsaint marys\b", "saint marys"),
        (r"\bmiami  fl\b", "miami fl"),
        (r"\bmiami  oh\b", "miami oh"),
    ]
    for pattern, replacement in exact_map:
        name = re.sub(pattern, replacement, name)

    # Ole Miss special case
    name = re.sub(r"\bmississippi\b(?!\s*st)", "ole miss", name)
    name = " ".join(name.split())
    return name

# ============================================================================
# LOAD ALL DATA SOURCES
# ============================================================================

print("[1/6] Loading data sources...")

# Master dataset (Kaggle historical data)
try:
    team_stats = pd.read_csv(PROCESSED_DIR / 'team_season_stats.csv')
    tourney_games = pd.read_csv(PROCESSED_DIR / 'tournament_games_features.csv')
    teams = pd.read_csv(KAGGLE_DIR / 'MTeams.csv')
    print(f"  ✓ Loaded master dataset: {len(team_stats)} team-seasons, {len(tourney_games)} tournament games")
except FileNotFoundError as e:
    print(f"  ✗ Master dataset not found! Run build_master_dataset.py first.")
    print(f"     Error: {e}")
    exit(1)

# External sources (optional but valuable)
try:
    f538 = pd.read_csv(DATA_DIR / 'fivethirtyeight_forecasts.csv')
    has_538 = True
    print(f"  ✓ Loaded 538 forecasts: {len(f538)} records")
except:
    has_538 = False
    print("  ! 538 forecasts not available (optional)")

try:
    torvik = pd.read_csv(DATA_DIR / 'barttorvik_historical.csv')
    # Handle potential header issues from data-pull
    if '1' in torvik.columns and 'team' not in torvik.columns:
        torvik = torvik.rename(columns={
            '0': 'rank', '1': 'team', '2': 'conf', '3': 'record', '4': 'adjoe',
            '5': 'oe_rank', '6': 'adjde', '7': 'de_rank', '8': 'barthag',
            '15': 'sos', '16': 'ncsos', '17': 'consos', '41': 'wab', '44': 'adjt'
        })
    has_torvik = True
    print(f"  ✓ Loaded Torvik  {len(torvik)} records")
except:
    has_torvik = False
    print("  ! Torvik data not available (optional)")

try:
    ncaa_net = pd.read_csv(DATA_DIR / 'ncaa_net.csv')
    has_net = True
    print(f"  ✓ Loaded NCAA NET: {len(ncaa_net)} teams")
except:
    has_net = False
    print("  ! NCAA NET not available (optional)")

print()

# ============================================================================
# CREATE TEAM NAME LOOKUP
# ============================================================================

print("[2/6] Building team name lookup...")

# Create mapping from TeamName to TeamID
teams['norm_name'] = teams['TeamName'].apply(normalize_name)
team_lookup = teams.set_index('norm_name')['TeamID'].to_dict()
team_id_to_name = teams.set_index('TeamID')['TeamName'].to_dict()

print(f"  ✓ Created lookup for {len(team_lookup)} teams")
print()

# ============================================================================
# BUILD ENHANCED TRAINING DATASET
# ============================================================================

print("[3/6] Building enhanced training dataset...")

# Start with tournament games as base
training_games = tourney_games[[
    'Season', 'WTeamID', 'LTeamID', 
    'SeedDiff', 'WinPctDiff', 'PointDiffDiff'
]].copy()

# Add Four Factors if available
if 'eFGDiff' in tourney_games.columns:
    training_games['eFGDiff'] = tourney_games['eFGDiff']
    training_games['TOV_RateDiff'] = tourney_games['TOV_RateDiff']
    print("  ✓ Added Four Factors differentials")

# Add Massey ratings if available
rating_cols = [c for c in tourney_games.columns if '_Rank_Diff' in c]
for col in rating_cols:
    training_games[col] = tourney_games[col]
if rating_cols:
    print(f"  ✓ Added {len(rating_cols)} external rating differentials")

print(f"  Initial training games: {len(training_games)}")

# Augment with Torvik data if available
if has_torvik:
    torvik['norm_name'] = torvik['team'].apply(normalize_name)
    torvik['season'] = pd.to_numeric(torvik.get('season', torvik.get('year', None)), errors='coerce')
    torvik = torvik.dropna(subset=['season'])
    torvik['season'] = torvik['season'].astype(int)
    
    # Convert all Torvik metrics to numeric (critical for calculations)
    torvik_metrics = ['adjoe', 'adjde', 'barthag', 'sos', 'wab', 'adjt']
    for col in torvik_metrics:
        if col in torvik.columns:
            torvik[col] = pd.to_numeric(torvik[col], errors='coerce')
    
    # Deduplicate
    torvik = torvik.drop_duplicates(subset=['season', 'norm_name'])
    
    # Join winner stats
    training_games = training_games.merge(
        teams[['TeamID', 'TeamName']],
        left_on='WTeamID',
        right_on='TeamID',
        how='left'
    )
    training_games['w_norm'] = training_games['TeamName'].apply(normalize_name)
    training_games = training_games.drop(['TeamID', 'TeamName'], axis=1)
    
    n_before = len(training_games)
    training_games = training_games.merge(
        torvik[['season', 'norm_name', 'adjoe', 'adjde', 'barthag', 'sos', 'wab', 'adjt']],
        left_on=['Season', 'w_norm'],
        right_on=['season', 'norm_name'],
        how='left',
        suffixes=('', '_w_torvik')
    )
    training_games = training_games.rename(columns={
        'adjoe': 'w_adjoe', 'adjde': 'w_adjde', 'barthag': 'w_barthag',
        'sos': 'w_sos', 'wab': 'w_wab', 'adjt': 'w_adjt'
    })
    
    # Join loser stats
    training_games = training_games.merge(
        teams[['TeamID', 'TeamName']],
        left_on='LTeamID',
        right_on='TeamID',
        how='left'
    )
    training_games['l_norm'] = training_games['TeamName'].apply(normalize_name)
    training_games = training_games.drop(['TeamID', 'TeamName'], axis=1)
    
    training_games = training_games.merge(
        torvik[['season', 'norm_name', 'adjoe', 'adjde', 'barthag', 'sos', 'wab', 'adjt']],
        left_on=['Season', 'l_norm'],
        right_on=['season', 'norm_name'],
        how='left',
        suffixes=('', '_l_torvik')
    )
    training_games = training_games.rename(columns={
        'adjoe': 'l_adjoe', 'adjde': 'l_adjde', 'barthag': 'l_barthag',
        'sos': 'l_sos', 'wab': 'l_wab', 'adjt': 'l_adjt'
    })
    
    # Calculate Torvik differentials (ensure numeric types)
    eps = 1e-6
    torvik_cols = ['adjoe', 'adjde', 'barthag', 'sos', 'wab', 'adjt']
    for col in torvik_cols:
        w_col = f'w_{col}'
        l_col = f'l_{col}'
        if w_col in training_games.columns and l_col in training_games.columns:
            # Ensure numeric types before operations
            training_games[w_col] = pd.to_numeric(training_games[w_col], errors='coerce')
            training_games[l_col] = pd.to_numeric(training_games[l_col], errors='coerce')
            
            training_games[f'{col}_diff'] = training_games[w_col] - training_games[l_col]
            training_games[f'{col}_ratio'] = training_games[w_col] / (training_games[l_col] + eps)
    
    n_with_torvik = training_games['w_adjoe'].notna().sum()
    print(f"  ✓ Augmented with Torvik: {n_with_torvik}/{len(training_games)} games matched ({100*n_with_torvik/len(training_games):.1f}%)")

# Create balanced dataset (both perspectives)
print("  Creating symmetric training samples...")

# Perspective 1: Winner (label = 1)
df1 = training_games.copy()
df1['team1_id'] = df1['WTeamID']
df1['team2_id'] = df1['LTeamID']
df1['label'] = 1

# Perspective 2: Loser (label = 0, flip differentials)
df2 = training_games.copy()
df2['team1_id'] = df2['LTeamID']
df2['team2_id'] = df2['WTeamID']
df2['label'] = 0

# Flip all differential features
diff_cols = [c for c in df2.columns if '_diff' in c.lower() or 'diff' in c]
for col in diff_cols:
    if col in df2.columns:
        df2[col] = -df2[col]

# Flip ratio features (inverse)
ratio_cols = [c for c in df2.columns if '_ratio' in c.lower()]
for col in ratio_cols:
    if col in df2.columns:
        df2[col] = 1.0 / (df2[col] + 1e-6)

# Select feature columns
feature_cols = ['Season', 'team1_id', 'team2_id'] + \
               [c for c in df1.columns if 'diff' in c.lower() or 'ratio' in c.lower()] + \
               ['label']

feature_cols = [c for c in feature_cols if c in df1.columns]

df1_clean = df1[feature_cols].copy()
df2_clean = df2[feature_cols].copy()

ml_training = pd.concat([df1_clean, df2_clean], ignore_index=True)
ml_training = ml_training.dropna(subset=['label'])

# Rename for consistency with original ETL
ml_training = ml_training.rename(columns={'Season': 'year', 'label': 'favorite_win_flag'})

print(f"  ✓ Created balanced dataset: {len(ml_training)} samples ({len(df1_clean)} + {len(df2_clean)})")
print(f"  ✓ Features: {[c for c in ml_training.columns if c not in ['year', 'team1_id', 'team2_id', 'favorite_win_flag']]}")
print()

# ============================================================================
# BUILD 2026 INFERENCE DATASET
# ============================================================================

print("[4/6] Building 2026 inference dataset...")

# Get 2026 tournament teams
if has_net:
    # Use NCAA NET top 68
    top_68 = ncaa_net.head(68).copy()
    top_68['norm_name'] = top_68['School'].apply(normalize_name)
    print(f"  Using NCAA NET top 68 teams")
else:
    # Fallback: use teams with good 2025 or 2026 stats
    recent_stats = team_stats[team_stats['Season'].isin([2025, 2026])]
    top_teams = recent_stats.nlargest(68, 'WinPct' if 'WinPct' in recent_stats.columns else 'Wins')
    top_68 = teams[teams['TeamID'].isin(top_teams['TeamID'])].copy()
    top_68['norm_name'] = top_68['TeamName'].apply(normalize_name)
    print(f"  Using top 68 teams from recent seasons (fallback)")

# Get 2026 team stats from master dataset
stats_2026 = team_stats[team_stats['Season'] == 2026].copy()

if len(stats_2026) == 0:
    print("  ! No 2026 stats found, using 2025 as fallback")
    stats_2026 = team_stats[team_stats['Season'] == 2025].copy()
    stats_2026['Season'] = 2026

# Add normalized names to stats for merging
stats_2026['norm_name'] = stats_2026['TeamID'].map(team_id_to_name).apply(normalize_name)

# Join team stats with tournament teams
current_teams = top_68.merge(
    stats_2026,
    on='norm_name',
    how='inner'
)

print(f"  ✓ Matched {len(current_teams)}/68 tournament teams with stats")

if len(current_teams) == 0:
    print("  ✗ ERROR: No teams matched! Check team name normalization.")
    print(f"  Sample top_68 names: {top_68['norm_name'].head(5).tolist()}")
    print(f"  Sample stats names: {stats_2026['norm_name'].head(5).tolist()}")
    exit(1)

# Augment with Torvik 2026 if available
if has_torvik:
    torvik_2026 = torvik[torvik['season'] == 2026].copy()
    if len(torvik_2026) == 0:
        torvik_2026 = torvik[torvik['season'] == 2025].copy()
        print("  ! Using 2025 Torvik data for 2026 inference")
    
    torvik_2026 = torvik_2026.drop_duplicates(subset=['norm_name'])
    
    n_before = len(current_teams)
    current_teams = current_teams.merge(
        torvik_2026[['norm_name', 'adjoe', 'adjde', 'barthag', 'sos', 'wab', 'adjt']],
        on='norm_name',
        how='left'
    )
    n_with_torvik = current_teams['adjoe'].notna().sum()
    print(f"  ✓ Augmented with Torvik: {n_with_torvik}/{len(current_teams)} teams")

# Generate all pairwise matchups
teams_list = current_teams.to_dict('records')
matchups = []
eps = 1e-6

for t1, t2 in itertools.combinations(teams_list, 2):
    matchup = {
        'team_a': t1.get('School', t1.get('TeamName', 'Unknown')),
        'team_b': t2.get('School', t2.get('TeamName', 'Unknown')),
        'team_a_id': t1.get('TeamID', -1),
        'team_b_id': t2.get('TeamID', -1),
    }
    
    # Master dataset features
    for feat in ['WinPct', 'PPG', 'OppPPG', 'PointDiff']:
        if feat in t1 and feat in t2:
            matchup[f'{feat}_diff'] = t1.get(feat, 0) - t2.get(feat, 0)
    
    # Four Factors
    for feat in ['eFG', 'TOV_Rate', 'ORB_Rate', 'FTA_Rate']:
        if feat in t1 and feat in t2:
            matchup[f'{feat}_diff'] = t1.get(feat, 0) - t2.get(feat, 0)
    
    # Seeds
    if 'SeedNum' in t1 and 'SeedNum' in t2:
        matchup['SeedNum_diff'] = t1.get('SeedNum', 8) - t2.get('SeedNum', 8)
    
    # Torvik features (if available)
    if has_torvik:
        for feat in ['adjoe', 'adjde', 'barthag', 'sos', 'wab', 'adjt']:
            t1_val = pd.to_numeric(t1.get(feat, np.nan), errors='coerce')
            t2_val = pd.to_numeric(t2.get(feat, np.nan), errors='coerce')
            if not pd.isna(t1_val) and not pd.isna(t2_val):
                matchup[f'{feat}_diff'] = t1_val - t2_val
                matchup[f'{feat}_ratio'] = t1_val / (t2_val + eps)
    
    matchups.append(matchup)

inference_df = pd.DataFrame(matchups)
inference_df = inference_df.dropna(subset=['team_a', 'team_b'])

print(f"  ✓ Created {len(inference_df)} pairwise matchups for 2026")
print(f"  ✓ Features per matchup: {len(inference_df.columns)}")
print()

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

print("[5/6] Saving datasets...")

ml_training.to_csv(DATA_DIR / 'ml_training_data_v2.csv', index=False)
print(f"  ✓ Saved: {DATA_DIR / 'ml_training_data_v2.csv'}")
print(f"    Shape: {ml_training.shape}")

inference_df.to_csv(DATA_DIR / 'ml_inference_data_2026_v2.csv', index=False)
print(f"  ✓ Saved: {DATA_DIR / 'ml_inference_data_2026_v2.csv'}")
print(f"    Shape: {inference_df.shape}")
print()

# ============================================================================
# DATA QUALITY SUMMARY
# ============================================================================

print("[6/6] Generating data quality summary...")

summary = {
    'training_samples': len(ml_training),
    'inference_matchups': len(inference_df),
    'seasons_covered': ml_training['year'].nunique(),
    'year_range': f"{ml_training['year'].min()}-{ml_training['year'].max()}",
    'features_training': len([c for c in ml_training.columns if c not in ['year', 'team1_id', 'team2_id', 'favorite_win_flag']]),
    'features_inference': len([c for c in inference_df.columns if c not in ['team_a', 'team_b', 'team_a_id', 'team_b_id']]),
    'missing_data_pct': {
        'training': {col: float(100 * ml_training[col].isna().sum() / len(ml_training)) 
                     for col in ml_training.columns if ml_training[col].isna().sum() > 0},
        'inference': {col: float(100 * inference_df[col].isna().sum() / len(inference_df)) 
                      for col in inference_df.columns if inference_df[col].isna().sum() > 0}
    },
    'data_sources': {
        'master_dataset': True,
        'fivethirtyeight': has_538,
        'torvik': has_torvik,
        'ncaa_net': has_net
    }
}

with open(DATA_DIR / 'etl_v2_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"  ✓ Saved: {DATA_DIR / 'etl_v2_summary.json'}")
print()

print("=" * 80)
print("ETL V2 COMPLETE!")
print("=" * 80)
print()
print("Summary:")
print(f"  Training samples:    {summary['training_samples']:,}")
print(f"  Inference matchups:  {summary['inference_matchups']:,}")
print(f"  Seasons covered:     {summary['seasons_covered']} ({summary['year_range']})")
print(f"  Features (train):    {summary['features_training']}")
print(f"  Features (infer):    {summary['features_inference']}")
print()
print("Data sources used:")
for source, available in summary['data_sources'].items():
    status = "✓" if available else "✗"
    print(f"  {status} {source}")
print()
print("Next steps:")
print("  1. Use ml_training_data_v2.csv for model training (all phases)")
print("  2. Use ml_inference_data_2026_v2.csv for 2026 predictions")
print("  3. Run phase2_baselines.py to establish performance benchmarks")
print()

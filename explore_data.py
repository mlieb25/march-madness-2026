#!/usr/bin/env python3
"""
March Madness Data Explorer

Quick data exploration and validation script.

Author: Mitchell Liebrecht
Date: March 15, 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MARCH MADNESS DATA EXPLORER")
print("=" * 80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path('data/march-machine-learning-mania-2026')
PROCESSED_DIR = Path('processed_data')

# ============================================================================
# 1. EXPLORE PROCESSED DATA (if available)
# ============================================================================

if (PROCESSED_DIR / 'team_season_stats.csv').exists():
    print("[1] TEAM SEASON STATISTICS")
    print("-" * 80)
    
    team_stats = pd.read_csv(PROCESSED_DIR / 'team_season_stats.csv')
    
    print(f"Dataset shape: {team_stats.shape}")
    print(f"Seasons covered: {team_stats['Season'].min()} - {team_stats['Season'].max()}")
    print(f"Total team-seasons: {len(team_stats):,}")
    print()
    
    print("Columns available:")
    for i, col in enumerate(team_stats.columns, 1):
        print(f"  {i:2d}. {col}")
    print()
    
    print("Sample: 2025 Tournament Teams (by seed)")
    tourney_2025 = team_stats[
        (team_stats['Season'] == 2025) & 
        (team_stats['SeedNum'].notna())
    ].sort_values('SeedNum')
    
    display_cols = ['TeamID', 'SeedNum', 'WinPct', 'PPG', 'OppPPG', 'PointDiff']
    if 'eFG' in tourney_2025.columns:
        display_cols.append('eFG')
    
    print(tourney_2025[display_cols].head(20).to_string(index=False))
    print()
    
    print("Missing value summary:")
    missing = team_stats.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    for col, count in missing.items():
        pct = 100 * count / len(team_stats)
        print(f"  {col:20s}: {count:6,} ({pct:5.1f}%)")
    print()
    
    print("-" * 80)
    print()

else:
    print("[!] Processed data not found. Run build_master_dataset.py first.")
    print()

# ============================================================================
# 2. EXPLORE TOURNAMENT GAMES
# ============================================================================

if (PROCESSED_DIR / 'tournament_games_features.csv').exists():
    print("[2] TOURNAMENT GAMES WITH FEATURES")
    print("-" * 80)
    
    tourney_games = pd.read_csv(PROCESSED_DIR / 'tournament_games_features.csv')
    
    print(f"Total tournament games: {len(tourney_games):,}")
    print(f"Seasons: {tourney_games['Season'].min()} - {tourney_games['Season'].max()}")
    print(f"Features per game: {len(tourney_games.columns)}")
    print()
    
    # Upset analysis
    print("Historical Upset Analysis (by seed differential):")
    upsets = tourney_games[tourney_games['SeedDiff'] > 0].copy()
    print(f"  Total upsets: {len(upsets):,} ({100*len(upsets)/len(tourney_games):.1f}% of games)")
    
    upset_summary = upsets.groupby('SeedDiff').size().sort_index()
    print("\n  Upsets by seed differential:")
    for diff, count in upset_summary.items():
        print(f"    {diff:2.0f} seed difference: {count:4} upsets")
    print()
    
    # Seed performance
    print("Tournament Performance by Seed:")
    
    # Create win/loss records by seed
    wins_by_seed = tourney_games.groupby('SeedNum_W').size()
    losses_by_seed = tourney_games.groupby('SeedNum_L').size()
    
    seed_performance = pd.DataFrame({
        'Seed': range(1, 17),
        'Wins': [wins_by_seed.get(s, 0) for s in range(1, 17)],
        'Losses': [losses_by_seed.get(s, 0) for s in range(1, 17)]
    })
    seed_performance['Games'] = seed_performance['Wins'] + seed_performance['Losses']
    seed_performance['WinPct'] = seed_performance['Wins'] / seed_performance['Games']
    seed_performance['AvgRounds'] = seed_performance['Wins'] / (tourney_games['Season'].nunique() * 4)
    
    print(seed_performance.to_string(index=False))
    print()
    
    print("-" * 80)
    print()

# ============================================================================
# 3. EXPLORE RAW DATA
# ============================================================================

print("[3] RAW DATA OVERVIEW")
print("-" * 80)

try:
    teams = pd.read_csv(DATA_DIR / 'MTeams.csv')
    print(f"Total teams in database: {len(teams):,}")
    print(f"  Active teams (D1): {len(teams[teams['LastD1Season'] >= 2025]):,}")
    print()
except:
    print("  Could not load teams data")
    print()

try:
    regular = pd.read_csv(DATA_DIR / 'MRegularSeasonCompactResults.csv')
    print(f"Regular season games: {len(regular):,}")
    print(f"  Seasons: {regular['Season'].min()} - {regular['Season'].max()}")
    print(f"  Games per season: {len(regular) / regular['Season'].nunique():.0f} avg")
    print()
except:
    print("  Could not load regular season data")
    print()

try:
    detailed = pd.read_csv(DATA_DIR / 'MRegularSeasonDetailedResults.csv')
    print(f"Detailed stats available: {len(detailed):,} games")
    print(f"  Seasons: {detailed['Season'].min()} - {detailed['Season'].max()}")
    print(f"  Box score columns: {list(detailed.columns[7:20])}")
    print()
except:
    print("  Could not load detailed stats")
    print()

try:
    massey = pd.read_csv(DATA_DIR / 'MMasseyOrdinals.csv')
    print(f"External ratings: {len(massey):,} team-day-system combinations")
    systems = massey['SystemName'].unique()
    print(f"  Rating systems: {len(systems)} ({', '.join(sorted(systems)[:10])}...)")
    print()
except:
    print("  External ratings not available")
    print()

print("-" * 80)
print()

# ============================================================================
# 4. DATA QUALITY CHECKS
# ============================================================================

if (PROCESSED_DIR / 'team_season_stats.csv').exists():
    print("[4] DATA QUALITY CHECKS")
    print("-" * 80)
    
    team_stats = pd.read_csv(PROCESSED_DIR / 'team_season_stats.csv')
    
    print("Checking for anomalies...")
    
    # Check win percentages
    bad_pcts = team_stats[(team_stats['WinPct'] < 0) | (team_stats['WinPct'] > 1)]
    print(f"  ✓ Win percentages in valid range [0, 1]: {len(bad_pcts) == 0}")
    
    # Check PPG
    low_ppg = team_stats[team_stats['PPG'] < 40]
    high_ppg = team_stats[team_stats['PPG'] > 120]
    print(f"  ✓ PPG in reasonable range [40, 120]: {len(low_ppg) + len(high_ppg) == 0}")
    if len(low_ppg) > 0:
        print(f"    ! {len(low_ppg)} teams with PPG < 40")
    if len(high_ppg) > 0:
        print(f"    ! {len(high_ppg)} teams with PPG > 120")
    
    # Check tournament seeds
    bad_seeds = team_stats[
        (team_stats['SeedNum'].notna()) & 
        ((team_stats['SeedNum'] < 1) | (team_stats['SeedNum'] > 16))
    ]
    print(f"  ✓ Tournament seeds in range [1, 16]: {len(bad_seeds) == 0}")
    
    # Check eFG% (if available)
    if 'eFG' in team_stats.columns:
        bad_efg = team_stats[
            (team_stats['eFG'].notna()) & 
            ((team_stats['eFG'] < 0.2) | (team_stats['eFG'] > 0.8))
        ]
        print(f"  ✓ eFG% in reasonable range [0.2, 0.8]: {len(bad_efg) == 0}")
        if len(bad_efg) > 0:
            print(f"    ! {len(bad_efg)} teams with unusual eFG%")
    
    print()
    print("Summary statistics (2025 season):")
    recent = team_stats[team_stats['Season'] == 2025]
    if len(recent) > 0:
        print(recent[['WinPct', 'PPG', 'OppPPG', 'PointDiff']].describe().to_string())
    else:
        print("  No 2025 data available yet")
    
    print()
    print("-" * 80)
    print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("EXPLORATION COMPLETE")
print("=" * 80)
print()
print("Recommended next steps:")

if not (PROCESSED_DIR / 'team_season_stats.csv').exists():
    print("  1. Run: python build_master_dataset.py")
    print("  2. Run: python build_prediction_features.py")
    print("  3. Run: python baseline_model.py")
else:
    print("  ✓ Processed data exists")
    if not (Path('predictions') / 'prediction_features_2026.csv').exists():
        print("  1. Run: python build_prediction_features.py")
        print("  2. Run: python baseline_model.py")
    else:
        print("  ✓ Prediction features exist")
        if not (Path('models') / 'baseline_results.csv').exists():
            print("  1. Run: python baseline_model.py")
        else:
            print("  ✓ Baseline models trained")
            print("  1. Explore advanced models (XGBoost, neural networks)")
            print("  2. Tune hyperparameters")
            print("  3. Create model ensemble")

print()

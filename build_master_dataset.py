#!/usr/bin/env python3
"""
March Madness Master Dataset Builder

This script combines all relevant data files into ML-ready datasets:
1. Team-season level features (aggregated statistics)
2. Game-level features for training
3. Pairwise matchup features for predictions

Author: Mitchell Liebrecht
Date: March 15, 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MARCH MADNESS MASTER DATASET BUILDER")
print("=" * 80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path('data/march-machine-learning-mania-2026')
OUTPUT_DIR = Path('processed_data')
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print()

# ============================================================================
# 1. LOAD ALL DATA FILES
# ============================================================================

print("[1/7] Loading data files...")

# Core files
teams = pd.read_csv(DATA_DIR / 'MTeams.csv')
seasons = pd.read_csv(DATA_DIR / 'MSeasons.csv')

# Tournament data
tourney_compact = pd.read_csv(DATA_DIR / 'MNCAATourneyCompactResults.csv')
tourney_detailed = pd.read_csv(DATA_DIR / 'MNCAATourneyDetailedResults.csv')
tourney_seeds = pd.read_csv(DATA_DIR / 'MNCAATourneySeeds.csv')

# Regular season data
regular_compact = pd.read_csv(DATA_DIR / 'MRegularSeasonCompactResults.csv')
regular_detailed = pd.read_csv(DATA_DIR / 'MRegularSeasonDetailedResults.csv')

# Supporting data
team_conferences = pd.read_csv(DATA_DIR / 'MTeamConferences.csv')
conferences = pd.read_csv(DATA_DIR / 'Conferences.csv')
try:
    massey_ordinals = pd.read_csv(DATA_DIR / 'MMasseyOrdinals.csv')
    has_massey = True
except:
    has_massey = False
    print("  Note: Massey ordinals not found, will skip external ratings")

print(f"  ✓ Loaded {len(teams)} teams")
print(f"  ✓ Loaded {len(tourney_compact)} tournament games")
print(f"  ✓ Loaded {len(regular_compact)} regular season games")
print(f"  ✓ Loaded {len(tourney_seeds)} tournament seeds")
print()

# ============================================================================
# 2. EXTRACT SEED INFORMATION
# ============================================================================

print("[2/7] Processing tournament seeds...")

def extract_seed_number(seed_str):
    """Extract numeric seed (1-16) from seed string like 'W01' or 'X16'"""
    if pd.isna(seed_str):
        return None
    return int(seed_str[1:3])

def extract_region(seed_str):
    """Extract region letter from seed string"""
    if pd.isna(seed_str):
        return None
    return seed_str[0]

tourney_seeds['SeedNum'] = tourney_seeds['Seed'].apply(extract_seed_number)
tourney_seeds['Region'] = tourney_seeds['Seed'].apply(extract_region)

print(f"  ✓ Processed seeds for {len(tourney_seeds)} team-seasons")
print()

# ============================================================================
# 3. BUILD TEAM-SEASON STATISTICS (REGULAR SEASON)
# ============================================================================

print("[3/7] Building team-season statistics...")

def calculate_team_season_stats(season_data):
    """Calculate comprehensive team statistics for each season"""
    
    # Combine winner and loser perspectives
    wins = season_data[['Season', 'WTeamID', 'WScore', 'LScore', 'WLoc']].copy()
    wins.columns = ['Season', 'TeamID', 'ScoreFor', 'ScoreAgainst', 'Loc']
    wins['Win'] = 1
    
    losses = season_data[['Season', 'LTeamID', 'LScore', 'WScore', 'WLoc']].copy()
    losses.columns = ['Season', 'TeamID', 'ScoreFor', 'ScoreAgainst', 'Loc']
    losses['Win'] = 0
    # Flip location for losses
    losses['Loc'] = losses['Loc'].map({'H': 'A', 'A': 'H', 'N': 'N'})
    
    # Combine all games
    all_games = pd.concat([wins, losses], ignore_index=True)
    
    # Calculate basic statistics
    team_stats = all_games.groupby(['Season', 'TeamID']).agg({
        'Win': ['sum', 'count'],
        'ScoreFor': 'mean',
        'ScoreAgainst': 'mean'
    }).reset_index()
    
    team_stats.columns = ['Season', 'TeamID', 'Wins', 'Games', 'PPG', 'OppPPG']
    team_stats['Losses'] = team_stats['Games'] - team_stats['Wins']
    team_stats['WinPct'] = team_stats['Wins'] / team_stats['Games']
    team_stats['PointDiff'] = team_stats['PPG'] - team_stats['OppPPG']
    
    # Home/Away/Neutral splits
    home_games = all_games[all_games['Loc'] == 'H'].groupby(['Season', 'TeamID'])['Win'].agg(['sum', 'count']).reset_index()
    home_games.columns = ['Season', 'TeamID', 'HomeWins', 'HomeGames']
    home_games['HomeWinPct'] = home_games['HomeWins'] / home_games['HomeGames']
    
    away_games = all_games[all_games['Loc'] == 'A'].groupby(['Season', 'TeamID'])['Win'].agg(['sum', 'count']).reset_index()
    away_games.columns = ['Season', 'TeamID', 'AwayWins', 'AwayGames']
    away_games['AwayWinPct'] = away_games['AwayWins'] / away_games['AwayGames']
    
    # Merge splits
    team_stats = team_stats.merge(home_games[['Season', 'TeamID', 'HomeWinPct']], 
                                   on=['Season', 'TeamID'], how='left')
    team_stats = team_stats.merge(away_games[['Season', 'TeamID', 'AwayWinPct']], 
                                   on=['Season', 'TeamID'], how='left')
    
    return team_stats

# Calculate for regular season
team_season_stats = calculate_team_season_stats(regular_compact)

print(f"  ✓ Created stats for {len(team_season_stats)} team-seasons")
print(f"  ✓ Features: Wins, Losses, WinPct, PPG, OppPPG, PointDiff, Home/Away splits")
print()

# ============================================================================
# 4. ADD ADVANCED STATISTICS (2003+)
# ============================================================================

print("[4/7] Adding advanced statistics from detailed data...")

def calculate_four_factors(detailed_data):
    """Calculate Four Factors and advanced metrics from detailed stats"""
    
    # Winner stats
    winner_stats = detailed_data[['Season', 'WTeamID']].copy()
    winner_stats['TeamID'] = detailed_data['WTeamID']
    winner_stats['eFG'] = (detailed_data['WFGM'] + 0.5 * detailed_data['WFGM3']) / detailed_data['WFGA']
    winner_stats['FTA_Rate'] = detailed_data['WFTA'] / detailed_data['WFGA']
    winner_stats['TOV_Rate'] = detailed_data['WTO'] / (detailed_data['WFGA'] + 0.44 * detailed_data['WFTA'] + detailed_data['WTO'])
    winner_stats['ORB_Rate'] = detailed_data['WOR'] / (detailed_data['WOR'] + detailed_data['LDR'])
    winner_stats['TRB'] = detailed_data['WOR'] + detailed_data['WDR']
    winner_stats['Ast'] = detailed_data['WAst']
    winner_stats['Stl'] = detailed_data['WStl']
    winner_stats['Blk'] = detailed_data['WBlk']
    winner_stats['PF'] = detailed_data['WPF']
    
    # Loser stats
    loser_stats = detailed_data[['Season', 'LTeamID']].copy()
    loser_stats['TeamID'] = detailed_data['LTeamID']
    loser_stats['eFG'] = (detailed_data['LFGM'] + 0.5 * detailed_data['LFGM3']) / detailed_data['LFGA']
    loser_stats['FTA_Rate'] = detailed_data['LFTA'] / detailed_data['LFGA']
    loser_stats['TOV_Rate'] = detailed_data['LTO'] / (detailed_data['LFGA'] + 0.44 * detailed_data['LFTA'] + detailed_data['LTO'])
    loser_stats['ORB_Rate'] = detailed_data['LOR'] / (detailed_data['LOR'] + detailed_data['WDR'])
    loser_stats['TRB'] = detailed_data['LOR'] + detailed_data['LDR']
    loser_stats['Ast'] = detailed_data['LAst']
    loser_stats['Stl'] = detailed_data['LStl']
    loser_stats['Blk'] = detailed_data['LBlk']
    loser_stats['PF'] = detailed_data['LPF']
    
    # Combine and aggregate
    all_stats = pd.concat([winner_stats, loser_stats], ignore_index=True)
    
    advanced_stats = all_stats.groupby(['Season', 'TeamID']).agg({
        'eFG': 'mean',
        'FTA_Rate': 'mean',
        'TOV_Rate': 'mean',
        'ORB_Rate': 'mean',
        'TRB': 'mean',
        'Ast': 'mean',
        'Stl': 'mean',
        'Blk': 'mean',
        'PF': 'mean'
    }).reset_index()
    
    return advanced_stats

advanced_stats = calculate_four_factors(regular_detailed)

# Merge with team_season_stats
team_season_stats = team_season_stats.merge(advanced_stats, 
                                             on=['Season', 'TeamID'], 
                                             how='left')

print(f"  ✓ Added Four Factors: eFG%, FTA_Rate, TOV_Rate, ORB_Rate")
print(f"  ✓ Added box score stats: TRB, Ast, Stl, Blk, PF")
print()

# ============================================================================
# 5. ADD TOURNAMENT SEEDS AND CONFERENCE INFO
# ============================================================================

print("[5/7] Merging seeds and conference data...")

# Add seeds
team_season_stats = team_season_stats.merge(
    tourney_seeds[['Season', 'TeamID', 'SeedNum', 'Region']], 
    on=['Season', 'TeamID'], 
    how='left'
)

# Add conference
team_season_stats = team_season_stats.merge(
    team_conferences[['Season', 'TeamID', 'ConfAbbrev']], 
    on=['Season', 'TeamID'], 
    how='left'
)

print(f"  ✓ Added tournament seeds (for teams that made tournament)")
print(f"  ✓ Added conference affiliations")
print()

# ============================================================================
# 6. ADD EXTERNAL RATINGS (MASSEY ORDINALS)
# ============================================================================

if has_massey:
    print("[6/7] Adding external ratings (Massey Ordinals)...")
    
    # Get the final rankings before tournament (day 133 is Selection Sunday)
    final_rankings = massey_ordinals[
        massey_ordinals['RankingDayNum'] == massey_ordinals.groupby(['Season', 'SystemName'])['RankingDayNum'].transform('max')
    ]
    
    # Pivot to get each rating system as a column
    rating_systems = ['POM', 'SAG', 'RPI', 'DOK', 'COL']  # Common systems
    
    for system in rating_systems:
        system_ratings = final_rankings[final_rankings['SystemName'] == system][['Season', 'TeamID', 'OrdinalRank']]
        system_ratings = system_ratings.rename(columns={'OrdinalRank': f'{system}_Rank'})
        team_season_stats = team_season_stats.merge(system_ratings, on=['Season', 'TeamID'], how='left')
    
    print(f"  ✓ Added {len(rating_systems)} rating systems: {', '.join(rating_systems)}")
    print()
else:
    print("[6/7] Skipping external ratings (data not available)")
    print()

# ============================================================================
# 7. CREATE TOURNAMENT GAME DATASET
# ============================================================================

print("[7/7] Building tournament game dataset...")

# Start with tournament games
tourney_games = tourney_compact.copy()

# Add team names
tourney_games = tourney_games.merge(
    teams[['TeamID', 'TeamName']], 
    left_on='WTeamID', 
    right_on='TeamID'
).rename(columns={'TeamName': 'WTeamName'}).drop('TeamID', axis=1)

tourney_games = tourney_games.merge(
    teams[['TeamID', 'TeamName']], 
    left_on='LTeamID', 
    right_on='TeamID'
).rename(columns={'TeamName': 'LTeamName'}).drop('TeamID', axis=1)

# Add winner stats
tourney_games = tourney_games.merge(
    team_season_stats,
    left_on=['Season', 'WTeamID'],
    right_on=['Season', 'TeamID'],
    how='left',
    suffixes=('', '_W')
).drop('TeamID', axis=1)

# Add loser stats
tourney_games = tourney_games.merge(
    team_season_stats,
    left_on=['Season', 'LTeamID'],
    right_on=['Season', 'TeamID'],
    how='left',
    suffixes=('_W', '_L')
).drop('TeamID', axis=1)

# Create differential features
tourney_games['SeedDiff'] = tourney_games['SeedNum_W'] - tourney_games['SeedNum_L']
tourney_games['WinPctDiff'] = tourney_games['WinPct_W'] - tourney_games['WinPct_L']
tourney_games['PointDiffDiff'] = tourney_games['PointDiff_W'] - tourney_games['PointDiff_L']

if 'eFG_W' in tourney_games.columns:
    tourney_games['eFGDiff'] = tourney_games['eFG_W'] - tourney_games['eFG_L']
    tourney_games['TOV_RateDiff'] = tourney_games['TOV_Rate_W'] - tourney_games['TOV_Rate_L']

print(f"  ✓ Created {len(tourney_games)} tournament game records")
print(f"  ✓ Added team statistics and differential features")
print()

# ============================================================================
# 8. SAVE DATASETS
# ============================================================================

print("Saving datasets...")

# Save team-season stats
team_season_stats.to_csv(OUTPUT_DIR / 'team_season_stats.csv', index=False)
print(f"  ✓ Saved: {OUTPUT_DIR / 'team_season_stats.csv'}")
print(f"    Shape: {team_season_stats.shape}")
print(f"    Columns: {list(team_season_stats.columns[:10])}...")
print()

# Save tournament games with features
tourney_games.to_csv(OUTPUT_DIR / 'tournament_games_features.csv', index=False)
print(f"  ✓ Saved: {OUTPUT_DIR / 'tournament_games_features.csv'}")
print(f"    Shape: {tourney_games.shape}")
print()

# Save a summary
summary = {
    'total_teams': len(teams),
    'total_seasons': len(seasons),
    'team_season_records': len(team_season_stats),
    'tournament_games': len(tourney_games),
    'regular_season_games': len(regular_compact),
    'features_in_team_stats': len(team_season_stats.columns),
    'seasons_with_detailed_stats': len(regular_detailed['Season'].unique()),
    'has_external_ratings': has_massey
}

with open(OUTPUT_DIR / 'dataset_summary.txt', 'w') as f:
    f.write("MARCH MADNESS MASTER DATASET SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    for key, value in summary.items():
        f.write(f"{key:.<40} {value}\n")

print(f"  ✓ Saved: {OUTPUT_DIR / 'dataset_summary.txt'}")
print()

print("=" * 80)
print("DATASET BUILDING COMPLETE!")
print("=" * 80)
print()
print("Available datasets:")
print(f"  1. team_season_stats.csv - {team_season_stats.shape[0]} rows × {team_season_stats.shape[1]} columns")
print(f"  2. tournament_games_features.csv - {tourney_games.shape[0]} rows × {tourney_games.shape[1]} columns")
print()
print("Sample team-season stats columns:")
for col in team_season_stats.columns[:15]:
    print(f"  • {col}")
print(f"  ... and {len(team_season_stats.columns) - 15} more")
print()
print("Next steps:")
print("  1. Load team_season_stats.csv for season-level analysis")
print("  2. Use tournament_games_features.csv for model training")
print("  3. Build pairwise features for 2026 predictions")
print()

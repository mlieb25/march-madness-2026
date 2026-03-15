import pandas as pd

f538 = pd.read_csv('data/fivethirtyeight_forecasts.csv')
torvik = pd.read_csv('data/barttorvik_historical.csv')

if '1' in torvik.columns and 'team' not in torvik.columns:
    torvik = torvik.rename(columns={
        '0': 'rank', '1': 'team', '2': 'conf', '3': 'record', '4': 'adjoe',
        '5': 'oe_rank', '6': 'adjde', '7': 'de_rank', '8': 'barthag',
        '15': 'sos', '16': 'ncsos', '17': 'consos', '41': 'wab', '44': 'season'
    })
    
print("Torvik columns after rename:")
print(torvik.columns.tolist()[:10], "... season column:", torvik['season'].unique() if 'season' in torvik.columns else "Missing")

from etl import normalize_name

games = f538[['year', 'favorite', 'underdog', 'favorite_win_flag']].dropna()
torvik['norm_name'] = torvik['team'].apply(normalize_name)
games['fav_norm'] = games['favorite'].apply(normalize_name)

print("Unique years in 538:", sorted(games['year'].unique()))
print("Unique seasons in Torvik:", sorted(torvik['season'].unique()) if 'season' in torvik.columns else "Missing season")

# Try merging without year
merged = games.merge(torvik[['norm_name', 'adjoe']], left_on='fav_norm', right_on='norm_name', how='inner')
print("Merge on name only:", merged.shape)

# Merging with year
merged_year = games.merge(torvik[['season', 'norm_name', 'adjoe']], left_on=['year', 'fav_norm'], right_on=['season', 'norm_name'], how='inner')
print("Merge on year + name:", merged_year.shape)


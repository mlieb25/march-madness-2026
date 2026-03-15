import pandas as pd

f538 = pd.read_csv('data/fivethirtyeight_forecasts.csv')
torvik = pd.read_csv('data/barttorvik_historical.csv')

def test_cols():
    print("Torvik columns default:", list(torvik.columns))
    
    torvik_rn = torvik.copy()
    if '1' in torvik_rn.columns and 'team' not in torvik_rn.columns:
        torvik_rn = torvik_rn.rename(columns={
            '0': 'rank', '1': 'team', '2': 'conf', '3': 'record', '4': 'adjoe',
            '5': 'oe_rank', '6': 'adjde', '7': 'de_rank', '8': 'barthag',
            '15': 'sos', '16': 'ncsos', '17': 'consos', '41': 'wab', '45': 'season'
        })
    print("Torvik columns renamed:")
    print([c for c in torvik_rn.columns if c in ['team', 'season', 'adjoe', 'adjde', 'barthag', 'sos', 'wab']])
    
    # Try finding the season column dynamically
    if 'season' not in torvik_rn.columns:
        for c in torvik_rn.columns:
            if torvik_rn[c].dtype in ['int64', 'float64']:
                uniques = getattr(torvik_rn, c).dropna().unique()
                if all(year in uniques for year in [2010, 2015, 2020]):
                    print(f"I found the season data in column: {c}")

test_cols()

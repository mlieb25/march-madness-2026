import pandas as pd

f538 = pd.read_csv('data/fivethirtyeight_forecasts.csv')
torvik = pd.read_csv('data/barttorvik_historical.csv')

def load_torvik():
    torvik_rn = torvik.copy()
    if '1' in torvik_rn.columns and 'team' not in torvik_rn.columns:
        torvik_rn = torvik_rn.rename(columns={
            '0': 'rank', '1': 'team', '2': 'conf', '3': 'record', '4': 'adjoe',
            '5': 'oe_rank', '6': 'adjde', '7': 'de_rank', '8': 'barthag',
            '15': 'sos', '16': 'ncsos', '17': 'consos', '41': 'wab'
        })
    return torvik_rn

torvik_rn = load_torvik()
print("f538 years:", f538['year'].unique())
print("f538 year dtype:", f538['year'].dtype)
if 'season' in torvik_rn.columns:
    print("Torvik seasons:", torvik_rn['season'].unique())
    print("Torvik season dtype:", torvik_rn['season'].dtype)
else:
    print("Torvik season column missing!")

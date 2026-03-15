import pandas as pd
f538 = pd.read_csv('data/fivethirtyeight_forecasts.csv')
torvik = pd.read_csv('data/barttorvik_historical.csv')
ncaa = pd.read_csv('data/ncaa_net.csv')

f538_teams = set(f538['favorite'].unique()) | set(f538['underdog'].unique())
torvik_teams = set(torvik['team'].unique())
ncaa_teams = set(ncaa['School'].unique())

print(f"538 unique teams: {len(f538_teams)}")
print(f"Torvik unique teams: {len(torvik_teams)}")
print(f"NCAA unique teams: {len(ncaa_teams)}")

# Find mismatches between 538 and Torvik
torvik_normalized = {t.lower().replace("st.", "state").replace(" ", ""): t for t in torvik_teams}

mismatches = []
for t in f538_teams:
    norm_t = t.lower().replace("st.", "state").replace(" ", "")
    if norm_t not in torvik_normalized and t not in torvik_teams:
        mismatches.append(t)

print(f"Mismatches between 538 and Torvik: {len(mismatches)}")
print(sorted(mismatches)[:20])

import pandas as pd

try:
    f538 = pd.read_csv('data/fivethirtyeight_forecasts.csv')
    torvik = pd.read_csv('data/barttorvik_historical.csv')
    ncaa = pd.read_csv('data/ncaa_net.csv')

    f538_teams = set(f538['favorite'].unique()) | set(f538['underdog'].unique())
    torvik_teams = set(torvik['team'].unique())
    ncaa_teams = set(ncaa['School'].unique())

    print(f"538 unique teams: {len(f538_teams)}")
    print(f"Torvik unique teams: {len(torvik_teams)}")
    print(f"NCAA unique teams: {len(ncaa_teams)}")

    # Simple normalization attempt
    def normalize(name):
        return str(name).lower().replace("st.", "state").replace(" ", "")

    torvik_normalized = {normalize(t): t for t in torvik_teams}
    ncaa_normalized = {normalize(t): t for t in ncaa_teams}

    mismatches = []
    for t in f538_teams:
        if normalize(t) not in torvik_normalized and t not in torvik_teams:
            mismatches.append(t)

    print(f"\nMismatches between 538 and Torvik: {len(mismatches)}")
    print(sorted(mismatches)[:30])
    
    mismatches_ncaa = []
    for t in ncaa_teams:
        if normalize(t) not in torvik_normalized and t not in torvik_teams:
            mismatches_ncaa.append(t)

    print(f"\nMismatches between NCAA NET and Torvik: {len(mismatches_ncaa)}")
    print(sorted(mismatches_ncaa)[:30])

except Exception as e:
    print(f"Error: {e}")

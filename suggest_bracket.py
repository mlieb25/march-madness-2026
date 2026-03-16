import pandas as pd
import numpy as np
import json
import os

# --- Config ---
ENSEMBLE_PROBS = "data/phase5_ensemble_probs.csv"
MATCHUPS_FILE = "matchups_2026.csv"
NET_FILE = "data/ncaa_net.csv"
OUTPUT_BRACKET = "suggested_bracket_2026.md"

def load_data():
    if not os.path.exists(ENSEMBLE_PROBS):
        raise FileNotFoundError(f"{ENSEMBLE_PROBS} not found.")
    if not os.path.exists(MATCHUPS_FILE):
        raise FileNotFoundError(f"{MATCHUPS_FILE} not found.")
    
    ensemble_df = pd.read_csv(ENSEMBLE_PROBS)
    matchups_df = pd.read_csv(MATCHUPS_FILE)
    
    net_df = None
    if os.path.exists(NET_FILE):
        net_df = pd.read_csv(NET_FILE)
    
    return ensemble_df, matchups_df, net_df

def build_prob_lookup(ensemble_df):
    lookup = {}
    for _, row in ensemble_df.iterrows():
        a, b = str(row['team_a']).strip(), str(row['team_b']).strip()
        p = float(row['ensemble_prob'])
        lookup[(a, b)] = p
        lookup[(b, a)] = 1 - p
    return lookup

def get_net_rank(team_name, net_df):
    if net_df is None: return 999
    # Try fuzzy matches or variations
    match = net_df[net_df['School'].str.lower().str.strip() == str(team_name).lower().strip()]
    if not match.empty:
        return int(match.iloc[0]['Rank'])
    return 999

def map_team_name(name):
    # Mapping from matchup screenshot scrapings to model/NET names
    mapping = {
        "Ohio State": "Ohio St.",
        "St John's": "St. John's (NY)",
        "CA Baptist": "California Baptist",
        "Michigan St": "Michigan St.",
        "Saint Mary's": "Saint Mary's (CA)",
        "Long Island": "LIU",
        "Hawai'i": "Hawaii",
        "Wright St": "Wright St.",
        "Kennesaw St": "Kennesaw St.",
        "South Florida": "South Fla.",
        "N Dakota St": "North Dakota St.",
        "Tennessee St": "Tennessee St.",
        "M-OH/SMU": "SMU",
        "PV/LEH": "Le Moyne",
        "UMBC/HOW": "Howard",
        "TEX/NCSU": "NC State",
        "Northern Iowa": "UNI",
        "UNI": "UNI",
        "California Baptist": "California Baptist",
    }
    return mapping.get(name, name)

def get_win_prob(t1, t2, lookup, net_df):
    # Map names
    m1, m2 = map_team_name(t1), map_team_name(t2)
    
    # Try lookup
    if (m1, m2) in lookup:
        return lookup[(m1, m2)]
    if (m2, m1) in lookup:
        return 1 - lookup[(m2, m1)]
    
    # Fallback 1: NET Rank
    r1 = get_net_rank(m1, net_df)
    r2 = get_net_rank(m2, net_df)
    
    if r1 < r2:
        return 0.90 # High confidence for rank advantage when model is missing
    elif r2 < r1:
        return 0.10
    
    return 0.5 # Coin flip

def main():
    ensemble_df, matchups_df, net_df = load_data()
    lookup = build_prob_lookup(ensemble_df)
    
    print("Building suggested bracket...")
    
    r64_results = []
    for _, row in matchups_df.iterrows():
        t1, t2 = row['team1'], row['team2']
        p1 = get_win_prob(t1, t2, lookup, net_df)
        
        winner = t1 if p1 >= 0.5 else t2
        conf = p1 if p1 >= 0.5 else 1 - p1
        
        r64_results.append({
            "region": row['region'],
            "matchup": f"{row['seed1']} {t1} vs {row['seed2']} {t2}",
            "winner": winner,
            "confidence": f"{conf:.1%}",
            "source": "Ensemble" if (map_team_name(t1), map_team_name(t2)) in lookup or (map_team_name(t2), map_team_name(t1)) in lookup else "NET Fallback"
        })
    
    with open(OUTPUT_BRACKET, "w") as f:
        f.write("# Suggested March Madness Bracket 2026 (Round of 64)\n\n")
        f.write("This bracket combines outputs from the Phase 5 Ensemble Model with NCAA NET ranking fallbacks for lower-seeded teams not included in the main inference set.\n\n")
        f.write("| Region | Matchup | Suggested Winner | Confidence | Source |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for res in r64_results:
            f.write(f"| {res['region']} | {res['matchup']} | **{res['winner']}** | {res['confidence']} | {res['source']} |\n")
            
    print(f"Bracket suggestion saved to {OUTPUT_BRACKET}")

if __name__ == "__main__":
    main()

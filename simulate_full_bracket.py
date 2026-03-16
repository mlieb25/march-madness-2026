import pandas as pd
import numpy as np
import os

# --- Config ---
ENSEMBLE_PROBS = "data/phase5_ensemble_probs.csv"
MATCHUPS_FILE = "matchups_2026.csv"
NET_FILE = "data/ncaa_net.csv"
OUTPUT_FILE = "full_bracket_picks_2026.md"

def load_data():
    ensemble_df = pd.read_csv(ENSEMBLE_PROBS)
    matchups_df = pd.read_csv(MATCHUPS_FILE)
    net_df = pd.read_csv(NET_FILE) if os.path.exists(NET_FILE) else None
    return ensemble_df, matchups_df, net_df

def build_prob_lookup(ensemble_df):
    lookup = {}
    for _, row in ensemble_df.iterrows():
        a, b = str(row['team_a']).strip(), str(row['team_b']).strip()
        p = float(row['ensemble_prob'])
        lookup[(a, b)] = p
        lookup[(b, a)] = 1 - p
    return lookup

def map_team_name(name):
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
    }
    return mapping.get(name, name)

def get_net_rank(team_name, net_df):
    if net_df is None: return 999
    match = net_df[net_df['School'].str.lower().str.strip() == str(team_name).lower().strip()]
    if not match.empty:
        return int(match.iloc[0]['Rank'])
    return 999

def get_win_prob(t1, t2, lookup, net_df):
    m1, m2 = map_team_name(t1), map_team_name(t2)
    if (m1, m2) in lookup: return lookup[(m1, m2)]
    if (m2, m1) in lookup: return 1 - lookup[(m2, m1)]
    r1, r2 = get_net_rank(m1, net_df), get_net_rank(m2, net_df)
    if r1 < r2: return 0.90
    if r2 < r1: return 0.10
    return 0.5

def simulate_game(t1, s1, t2, s2, lookup, net_df):
    p1 = get_win_prob(t1, t2, lookup, net_df)
    winner = (t1, s1) if p1 >= 0.5 else (t2, s2)
    loser = (t2, s2) if p1 >= 0.5 else (t1, s1)
    prob = p1 if p1 >= 0.5 else 1 - p1
    return winner, loser, prob

def main():
    ensemble_df, matchups_df, net_df = load_data()
    lookup = build_prob_lookup(ensemble_df)
    
    rounds = {
        "Round of 64": [],
        "Round of 32": [],
        "Sweet 16": [],
        "Elite 8": [],
        "Final Four": [],
        "Championship": []
    }
    
    # Regional Simulation
    regional_winners = {} # region -> list of (team, seed)
    
    for region in ["East", "South", "West", "Midwest"]:
        reg_df = matchups_df[matchups_df['region'] == region]
        r64_teams = []
        for _, row in reg_df.iterrows():
            r64_teams.append(((row['team1'], row['seed1']), (row['team2'], row['seed2'])))
        
        # R64 -> R32
        r32_teams = []
        for t1_info, t2_info in r64_teams:
            winner, loser, prob = simulate_game(t1_info[0], t1_info[1], t2_info[0], t2_info[1], lookup, net_df)
            rounds["Round of 64"].append(f"{region}: ({t1_info[1]}){t1_info[0]} vs ({t2_info[1]}){t2_info[0]} -> **{winner[0]}** ({prob:.1%})")
            r32_teams.append(winner)
        
        # R32 -> S16
        s16_teams = []
        for i in range(0, 8, 2):
            t1_info, t2_info = r32_teams[i], r32_teams[i+1]
            winner, loser, prob = simulate_game(t1_info[0], t1_info[1], t2_info[0], t2_info[1], lookup, net_df)
            rounds["Round of 32"].append(f"{region}: ({t1_info[1]}){t1_info[0]} vs ({t2_info[1]}){t2_info[0]} -> **{winner[0]}** ({prob:.1%})")
            s16_teams.append(winner)
            
        # S16 -> E8
        e8_teams = []
        for i in range(0, 4, 2):
            t1_info, t2_info = s16_teams[i], s16_teams[i+1]
            winner, loser, prob = simulate_game(t1_info[0], t1_info[1], t2_info[0], t2_info[1], lookup, net_df)
            rounds["Sweet 16"].append(f"{region}: ({t1_info[1]}){t1_info[0]} vs ({t2_info[1]}){t2_info[0]} -> **{winner[0]}** ({prob:.1%})")
            e8_teams.append(winner)
            
        # E8 -> FF
        t1_info, t2_info = e8_teams[0], e8_teams[1]
        winner, loser, prob = simulate_game(t1_info[0], t1_info[1], t2_info[0], t2_info[1], lookup, net_df)
        rounds["Elite 8"].append(f"{region} Final: ({t1_info[1]}){t1_info[0]} vs ({t2_info[1]}){t2_info[0]} -> **{winner[0]}** ({prob:.1%})")
        regional_winners[region] = winner

    # Final Four (Correction: East vs South, West vs Midwest)
    ff_matchups = [("East", "South"), ("West", "Midwest")]
    finalists = []
    for r1, r2 in ff_matchups:
        t1_info, t2_info = regional_winners[r1], regional_winners[r2]
        winner, loser, prob = simulate_game(t1_info[0], t1_info[1], t2_info[0], t2_info[1], lookup, net_df)
        rounds["Final Four"].append(f"{r1} vs {r2}: ({t1_info[1]}){t1_info[0]} vs ({t2_info[1]}){t2_info[0]} -> **{winner[0]}** ({prob:.1%})")
        finalists.append(winner)
    
    # Championship
    t1_info, t2_info = finalists[0], finalists[1]
    winner, loser, prob = simulate_game(t1_info[0], t1_info[1], t2_info[0], t2_info[1], lookup, net_df)
    rounds["Championship"].append(f"National Final: ({t1_info[1]}){t1_info[0]} vs ({t2_info[1]}){t2_info[0]} -> **{winner[0]}** ({prob:.1%})")

    # Output to Markdown
    with open(OUTPUT_FILE, "w") as f:
        f.write("# Full Bracket Picks 2026\n\n")
        f.write("Generated using Phase 5 Ensemble predictions and NCAA NET fallbacks. Selection strategy: **Chalk (Highest Probability)**.\n\n")
        
        for rnd, picks in rounds.items():
            f.write(f"## {rnd}\n")
            for pick in picks:
                f.write(f"- {pick}\n")
            f.write("\n")
            
        f.write(f"\n### Predicted Champion: **{winner[0]}**")
        
    print(f"Full bracket picks saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

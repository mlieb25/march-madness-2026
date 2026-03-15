import pandas as pd
import os
import itertools
import numpy as np
import re

def normalize_name(name):
    """Normalize team names to maximize join hits.

    FIX (2026-03-15): Replaced sequential str.replace dict with a two-pass
    approach using regex word boundaries.  The old approach had cascading
    collisions, e.g.:
      - "North Carolina State" → "unc" fired before NC State check → "unc st"
      - "Mississippi State"   → "ole miss" fired before MS State check → "ole miss st"
    """
    if pd.isna(name):
        return ""
    name = str(name).lower().strip()

    # ── Pass 1: punctuation / symbol cleanup ──────────────────────────────────
    name = name.replace("'", "").replace("-", " ").replace("&", "and")
    name = name.replace("(", "").replace(")", "")

    # Expand "st." abbreviation with word-boundary guard
    name = re.sub(r"\bst\.", "st", name)

    # Normalize " state" suffix → " st"
    name = name.replace(" state", " st")

    # Strip trailing university / univ
    name = re.sub(r"\buniversity$", "", name).strip()
    name = re.sub(r"\buniv$", "", name).strip()

    # ── Pass 2: specific team mappings (post-normalization) ───────────────────
    # Ordered so longer / more-specific patterns precede their substrings.
    # "north carolina st" must precede "north carolina" → "unc".
    # "mississippi" uses negative lookahead so "mississippi st" is preserved.
    exact_map = [
        (r"\bnorth carolina st\b",     "nc st"),       # NC State
        (r"\bnorth carolina\b",        "unc"),          # UNC Chapel Hill
        (r"\blouisiana st\b",          "lsu"),
        (r"\bconnecticut\b",           "uconn"),
        (r"\bsouthern california\b",   "usc"),
        (r"\bcentral florida\b",       "ucf"),
        (r"\bsouthern methodist\b",    "smu"),
        (r"\btexas christian\b",       "tcu"),
        (r"\bmassachusetts\b",         "umass"),
        (r"\bpennsylvania\b",          "penn"),
        (r"\bbrigham young\b",         "byu"),
        (r"\bvirginia commonwealth\b", "vcu"),
        (r"\bstephen f austin\b",      "stephen f austin"),
        (r"\bsaint marys  ca\b",       "saint marys"),
        (r"\bsaint marys\b",           "saint marys"),
        (r"\bmiami  fl\b",             "miami fl"),
        (r"\bmiami  oh\b",             "miami oh"),
    ]
    for pattern, replacement in exact_map:
        name = re.sub(pattern, replacement, name)

    # Ole Miss: only when "mississippi" is NOT followed by " st"
    name = re.sub(r"\bmississippi\b(?!\s*st)", "ole miss", name)

    # Collapse any double-spaces introduced by removals
    name = " ".join(name.split())
    return name

def load_data():
    """Load all necessary CSV files."""
    try:
        f538 = pd.read_csv('data/fivethirtyeight_forecasts.csv')
        # Torvik historical has no header in the original CSV, but we added it in data-pull by appending rows.
        # However, the first row of data-pull might just be index if we didn't carefully extract the header.
        # Let's read and see. If it has 'team' column, we use it. If it has '1' column, we rename.
        torvik = pd.read_csv('data/barttorvik_historical.csv')
        
        # If columns are string integers from data-pull header=None logic
        if '1' in torvik.columns and 'team' not in torvik.columns:
            torvik = torvik.rename(columns={
                '0': 'rank', '1': 'team', '2': 'conf', '3': 'record', '4': 'adjoe',
                '5': 'oe_rank', '6': 'adjde', '7': 'de_rank', '8': 'barthag',
                '15': 'sos', '16': 'ncsos', '17': 'consos', '41': 'wab', '44': 'adjt'
            })
            
        ncaa = pd.read_csv('data/ncaa_net.csv')
        return f538, torvik, ncaa
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def build_training_data(f538, torvik):
    """Build the ML training dataset from historical matchups."""
    print("Building training dataset...")
    
    # We only want actual tournament games, not just initial forecasts.
    # 538 dataset contains round-by-round forecasts. We can just use the matchups and win flags.
    # To avoid data leakage, we drop duplicates of (year, favorite, underdog).
    games = f538[['year', 'favorite', 'underdog', 'favorite_win_flag']].dropna().drop_duplicates(subset=['year', 'favorite', 'underdog'])
    
    # Normalize names
    torvik['norm_name'] = torvik['team'].apply(normalize_name)
    games['fav_norm'] = games['favorite'].apply(normalize_name)
    games['und_norm'] = games['underdog'].apply(normalize_name)
    
    # Deduplicate torvik to ensure uniqueness per team per year
    torvik = torvik.drop_duplicates(subset=['season', 'norm_name'])
    
    n_games_before_join = len(games)
    # Merge for favorite
    merged = games.merge(
        torvik[['season', 'norm_name', 'adjoe', 'adjde', 'barthag', 'sos', 'wab', 'adjt']], 
        left_on=['year', 'fav_norm'], right_on=['season', 'norm_name'], how='inner'
    )
    n_after_fav = len(merged)
    print(f"  Join (games → Torvik favorite): {n_games_before_join} → {n_after_fav} rows (dropped {n_games_before_join - n_after_fav})")
    merged = merged.rename(columns={'adjoe': 'fav_adjoe', 'adjde': 'fav_adjde', 'barthag': 'fav_barthag', 'sos': 'fav_sos', 'wab': 'fav_wab', 'adjt': 'fav_adjt'})

    # Merge for underdog
    merged = merged.merge(
        torvik[['season', 'norm_name', 'adjoe', 'adjde', 'barthag', 'sos', 'wab', 'adjt']],
        left_on=['year', 'und_norm'], right_on=['season', 'norm_name'], how='inner'
    )
    n_after_und = len(merged)
    print(f"  Join (games → Torvik underdog): {n_after_fav} → {n_after_und} rows (dropped {n_after_fav - n_after_und})")
    merged = merged.rename(columns={'adjoe': 'und_adjoe', 'adjde': 'und_adjde', 'barthag': 'und_barthag', 'sos': 'und_sos', 'wab': 'und_wab', 'adjt': 'und_adjt'})
    
    # Calculate differentials
    eps = 1e-6
    merged['adjoe_diff'] = pd.to_numeric(merged['fav_adjoe'], errors='coerce') - pd.to_numeric(merged['und_adjoe'], errors='coerce')
    merged['adjoe_ratio'] = pd.to_numeric(merged['fav_adjoe'], errors='coerce') / (pd.to_numeric(merged['und_adjoe'], errors='coerce') + eps)
    
    merged['adjde_diff'] = pd.to_numeric(merged['fav_adjde'], errors='coerce') - pd.to_numeric(merged['und_adjde'], errors='coerce')
    merged['adjde_ratio'] = pd.to_numeric(merged['fav_adjde'], errors='coerce') / (pd.to_numeric(merged['und_adjde'], errors='coerce') + eps)
    
    merged['barthag_diff'] = pd.to_numeric(merged['fav_barthag'], errors='coerce') - pd.to_numeric(merged['und_barthag'], errors='coerce')
    merged['barthag_ratio'] = pd.to_numeric(merged['fav_barthag'], errors='coerce') / (pd.to_numeric(merged['und_barthag'], errors='coerce') + eps)
    
    merged['sos_diff'] = pd.to_numeric(merged['fav_sos'], errors='coerce') - pd.to_numeric(merged['und_sos'], errors='coerce')
    merged['sos_ratio'] = pd.to_numeric(merged['fav_sos'], errors='coerce') / (pd.to_numeric(merged['und_sos'], errors='coerce') + eps)
    
    merged['wab_diff'] = pd.to_numeric(merged['fav_wab'], errors='coerce') - pd.to_numeric(merged['und_wab'], errors='coerce')
    merged['wab_ratio'] = pd.to_numeric(merged['fav_wab'], errors='coerce') / (pd.to_numeric(merged['und_wab'], errors='coerce') + eps)
    
    merged['adjt_diff'] = pd.to_numeric(merged['fav_adjt'], errors='coerce') - pd.to_numeric(merged['und_adjt'], errors='coerce')
    merged['adjt_ratio'] = pd.to_numeric(merged['fav_adjt'], errors='coerce') / (pd.to_numeric(merged['und_adjt'], errors='coerce') + eps)
    
    # Drop NAs
    merged = merged.dropna(subset=['adjoe_diff', 'favorite_win_flag'])
    
    target = 'favorite_win_flag'
    diff_feats = ['adjoe_diff', 'adjde_diff', 'barthag_diff', 'sos_diff', 'wab_diff', 'adjt_diff']
    ratio_feats = ['adjoe_ratio', 'adjde_ratio', 'barthag_ratio', 'sos_ratio', 'wab_ratio', 'adjt_ratio']
    features = diff_feats + ratio_feats
    
    final_df = merged[['year', 'favorite', 'underdog'] + features + [target]]
    
    # Also add symmetric inverse rows so the model doesn't overfit to "favorite" being team A
    inverse_df = merged.copy()
    inverse_df = inverse_df.rename(columns={'favorite': 'underdog', 'underdog': 'favorite'})
    for feat in diff_feats:
        inverse_df[feat] = -inverse_df[feat]
    for feat in ratio_feats:
        # if A/B was ratio, inverse is B/A which is 1 / ratio
        inverse_df[feat] = 1.0 / (inverse_df[feat] + eps)
        
    inverse_df['favorite_win_flag'] = 1 - inverse_df['favorite_win_flag']
    inverse_df = inverse_df[['year', 'favorite', 'underdog'] + features + [target]]
    
    final_train = pd.concat([final_df, inverse_df]).reset_index(drop=True)

    # Symmetry assertion: for each row there must be an inverse with negated diffs, inverted ratios, flipped label
    eps = 1e-6
    assert len(final_train) == 2 * len(final_df), "Expected exactly 2x rows (original + inverse)"
    for idx in range(min(100, len(final_df))):  # sample up to 100 rows
        row = final_df.iloc[idx]
        inv = final_train.iloc[len(final_df) + idx]
        assert inv["favorite"] == row["underdog"] and inv["underdog"] == row["favorite"]
        assert inv["favorite_win_flag"] == 1 - row["favorite_win_flag"]
        for f in diff_feats:
            np.testing.assert_approx_equal(inv[f], -row[f], significant=5)
        for f in ratio_feats:
            np.testing.assert_approx_equal(inv[f], 1.0 / (row[f] + eps), significant=5)
    print("  Symmetry assertion passed (sample of rows).")
    
    print(f"Training data shape: {final_train.shape}")
    final_train.to_csv("data/ml_training_data.csv", index=False)
    return final_train

def build_inference_data(torvik, ncaa):
    """Build the ML inference dataset for 2026 matchups."""
    print("Building 2026 inference dataset...")
    
    # Get top 68 teams from NCAA NET
    top_68 = ncaa.head(68).copy()
    top_68['norm_name'] = top_68['School'].apply(normalize_name)
    
    # Get 2026 Torvik stats
    torvik_2026 = torvik[torvik['season'] == 2026].copy()
    if torvik_2026.empty:
        # If 2026 Torvik is entirely completely missing, trying 2025 as fallback for testing 
        # (Though data-pull should have pulled it)
        print("Warning: No 2026 data in torvik historical. Falling back to 2025 for inference test.")
        torvik_2026 = torvik[torvik['season'] == 2025].copy()
        
    torvik_2026['norm_name'] = torvik_2026['team'].apply(normalize_name)
    torvik_2026 = torvik_2026.drop_duplicates(subset=['norm_name'])
    
    # Join NET teams with Torvik stats
    n_net = len(top_68)
    current_teams = top_68.merge(
        torvik_2026[['norm_name', 'adjoe', 'adjde', 'barthag', 'sos', 'wab', 'adjt']],
        on='norm_name', how='inner'
    )
    print(f"  Inference join (NET → Torvik 2026): {n_net} → {len(current_teams)} teams (dropped {n_net - len(current_teams)}).")
    print(f"Successfully matched {len(current_teams)} out of 68 teams for 2026.")
    
    # Generate all pairwise matchups combination
    teams = current_teams.to_dict('records')
    matchups = []
    eps = 1e-6
    
    for t1, t2 in itertools.combinations(teams, 2):
        t1_oe = pd.to_numeric(t1['adjoe'], errors='coerce')
        t2_oe = pd.to_numeric(t2['adjoe'], errors='coerce')
        t1_de = pd.to_numeric(t1['adjde'], errors='coerce')
        t2_de = pd.to_numeric(t2['adjde'], errors='coerce')
        t1_b = pd.to_numeric(t1['barthag'], errors='coerce')
        t2_b = pd.to_numeric(t2['barthag'], errors='coerce')
        t1_s = pd.to_numeric(t1['sos'], errors='coerce')
        t2_s = pd.to_numeric(t2['sos'], errors='coerce')
        t1_w = pd.to_numeric(t1['wab'], errors='coerce')
        t2_w = pd.to_numeric(t2['wab'], errors='coerce')
        t1_t = pd.to_numeric(t1['adjt'], errors='coerce')
        t2_t = pd.to_numeric(t2['adjt'], errors='coerce')
        
        matchups.append({
            'team_a': t1['School'],
            'team_b': t2['School'],
            'adjoe_diff': t1_oe - t2_oe,
            'adjde_diff': t1_de - t2_de,
            'barthag_diff': t1_b - t2_b,
            'sos_diff': t1_s - t2_s,
            'wab_diff': t1_w - t2_w,
            'adjt_diff': t1_t - t2_t,
            'adjoe_ratio': t1_oe / (t2_oe + eps),
            'adjde_ratio': t1_de / (t2_de + eps),
            'barthag_ratio': t1_b / (t2_b + eps),
            'sos_ratio': t1_s / (t2_s + eps),
            'wab_ratio': t1_w / (t2_w + eps),
            'adjt_ratio': t1_t / (t2_t + eps)
        })
        
    inference_df = pd.DataFrame(matchups)
    inference_df = inference_df.dropna()
    print(f"Inference data shape: {inference_df.shape}")
    inference_df.to_csv("data/ml_inference_data_2026.csv", index=False)
    return inference_df

if __name__ == "__main__":
    f538, torvik, ncaa = load_data()
    if f538 is not None:
        build_training_data(f538, torvik)
        build_inference_data(torvik, ncaa)
        print("ETL process complete.")

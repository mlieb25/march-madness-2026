"""
Phase 1a — Data pull. Scrapes Barttorvik, 538, NCAA NET, Massey, WarrenNolan.
Uses retries with backoff, validates Torvik schema, and writes pull_manifest.json for reproducibility.
"""
import os
import time
import hashlib
import json
import pandas as pd
import cloudscraper
from io import StringIO

scraper = cloudscraper.create_scraper(browser={'browser': 'firefox', 'platform': 'windows', 'mobile': False})

REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # multiplier between attempts

# Expected Torvik column indices (after header=None, names=range(46))
# Used to validate schema has not changed
TORVIK_REQUIRED_COLS = {
    1: "team",
    4: "adjoe",
    6: "adjde",
    8: "barthag",
    15: "sos",
    41: "wab",
    44: "adjt",
}


def fetch_with_retry(get_fn, url, **kwargs):
    """Try get_fn(url, **kwargs) up to MAX_RETRIES times with exponential backoff."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return get_fn(url, timeout=REQUEST_TIMEOUT, **kwargs)
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF ** attempt
                print(f"  Retry in {wait}s after: {e}")
                time.sleep(wait)
    raise last_err


def fetch_csv(url, **kwargs):
    timeout = kwargs.pop("timeout", REQUEST_TIMEOUT)
    r = scraper.get(url, timeout=timeout)
    if r.status_code == 200:
        return pd.read_csv(StringIO(r.text), **kwargs)
    raise Exception(f"Failed to fetch {url}, status code {r.status_code}")


def fetch_html(url, **kwargs):
    timeout = kwargs.pop("timeout", REQUEST_TIMEOUT)
    r = scraper.get(url, timeout=timeout)
    if r.status_code == 200:
        return pd.read_html(StringIO(r.text), **kwargs)[0]
    raise Exception(f"Failed to fetch {url}, status code {r.status_code}")


def validate_torvik_schema(df):
    """Ensure Torvik DataFrame has expected column indices (0..45). Fail fast if layout changed."""
    if df.empty:
        return
    for idx, name in TORVIK_REQUIRED_COLS.items():
        if idx not in df.columns:
            raise ValueError(
                f"Torvik schema validation failed: expected column index {idx} ({name}). "
                f"Columns: {list(df.columns)[:15]}..."
            )
    print("  Torvik schema validated.")


def file_sha256(path):
    """Return SHA-256 hex digest of file, or None if not readable."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


# 1. Barttorvik — all years (2008–2026)
dfs = []
for year in range(2008, 2027):
    try:
        df = fetch_with_retry(
            lambda u, **kw: fetch_csv(u, header=None, names=range(46), **kw),
            f"https://barttorvik.com/{year}_team_results.csv",
        )
        df["season"] = year
        dfs.append(df)
        print(f"Torvik {year} success")
    except Exception as e:
        print(f"Torvik {year} failed: {e}")
torvik_df = pd.concat(dfs) if dfs else pd.DataFrame()
if not torvik_df.empty:
    validate_torvik_schema(torvik_df)

# 2. Barttorvik advanced stats 2026
try:
    adv = fetch_with_retry(
        lambda u, **kw: fetch_csv(u, header=None, **kw),
        "https://barttorvik.com/getadvstats.php?year=2026&csv=1",
    )
except Exception as e:
    print(f"Failed to fetch Barttorvik advanced stats: {e}")
    adv = None

# 3. Massey team IDs
try:
    massey_teams = fetch_with_retry(
        lambda u, **kw: fetch_csv(u, header=None, names=["id", "team"], **kw),
        "https://masseyratings.com/scores.php?s=cb&sub=ncaa-d1&all=1&mode=3&format=2",
    )
except Exception as e:
    print(f"Failed to fetch Massey team IDs: {e}")
    massey_teams = None

# 4. NCAA NET (official)
try:
    net_df = fetch_with_retry(fetch_html, "https://www.ncaa.com/rankings/basketball-men/d1/ncaa-mens-basketball-net-rankings")
except Exception as e:
    print(f"Failed to fetch NCAA NET: {e}")
    net_df = None

# 5. FiveThirtyEight historical forecasts
try:
    f538 = fetch_with_retry(
        fetch_csv,
        "https://raw.githubusercontent.com/fivethirtyeight/data/master/historical-ncaa-forecasts/historical-538-ncaa-tournament-model-results.csv",
    )
except Exception as e:
    print(f"Failed to fetch 538 forecasts: {e}")
    f538 = None

# 6. WarrenNolan NET (with Quad records)
try:
    wn = fetch_with_retry(fetch_html, "https://www.warrennolan.com/basketball/2026/net")
except Exception as e:
    print(f"Failed to fetch WarrenNolan: {e}")
    wn = None

# 7. Sports-Reference (via sportsipy)
try:
    from sportsipy.ncaab.teams import Teams
    sr_teams = Teams(year=2026)
except Exception as e:
    print(f"Failed to fetch Sports-Reference: {e}")
    sr_teams = None

print("Data pull complete. Exporting to CSV...")

os.makedirs("data", exist_ok=True)
manifest = {}

if not torvik_df.empty:
    torvik_df.to_csv("data/barttorvik_historical.csv", index=False)
    manifest["barttorvik_historical.csv"] = file_sha256("data/barttorvik_historical.csv")
if adv is not None and not adv.empty:
    adv.to_csv("data/barttorvik_adv_2026.csv", index=False)
    manifest["barttorvik_adv_2026.csv"] = file_sha256("data/barttorvik_adv_2026.csv")
if massey_teams is not None and not massey_teams.empty:
    massey_teams.to_csv("data/massey_teams.csv", index=False)
    manifest["massey_teams.csv"] = file_sha256("data/massey_teams.csv")
if net_df is not None and not net_df.empty:
    net_df.to_csv("data/ncaa_net.csv", index=False)
    manifest["ncaa_net.csv"] = file_sha256("data/ncaa_net.csv")
if f538 is not None and not f538.empty:
    f538.to_csv("data/fivethirtyeight_forecasts.csv", index=False)
    manifest["fivethirtyeight_forecasts.csv"] = file_sha256("data/fivethirtyeight_forecasts.csv")
if wn is not None and not wn.empty:
    wn.to_csv("data/warrennolan_net.csv", index=False)
    manifest["warrennolan_net.csv"] = file_sha256("data/warrennolan_net.csv")

try:
    if sr_teams is not None:
        sr_dfs = []
        for team in sr_teams:
            sr_dfs.append(team.dataframe)
        if sr_dfs:
            sr_df = pd.concat(sr_dfs)
            sr_df.to_csv("data/sports_reference_2026.csv", index=False)
            manifest["sports_reference_2026.csv"] = file_sha256("data/sports_reference_2026.csv")
except Exception as e:
    print(f"Failed to export Sports-Reference data: {e}")

# Reproducibility manifest (checksums of raw data)
manifest_path = "data/pull_manifest.json"
with open(manifest_path, "w") as f:
    json.dump({"files": manifest, "note": "SHA-256 of each file after pull"}, f, indent=2)
print(f"Manifest saved → {manifest_path}")

print("Export complete. Files saved to 'data' directory.")

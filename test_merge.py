import pandas as pd

torvik = pd.read_csv("data/barttorvik_historical.csv")
print("Torvik columns:", torvik.columns.tolist()[:10], "... (total: {})".format(len(torvik.columns)))

try:
    adv = pd.read_csv("data/barttorvik_adv_2026.csv")
    print("Adv stats 2026 shape:", adv.shape)
    print("Adv stats 2026 columns:", adv.columns.tolist()[:10])
except Exception as e:
    print("Adv stats error:", e)
    
massey = pd.read_csv("data/massey_teams.csv")
print("Massey teams shape:", massey.shape)

ncaa = pd.read_csv("data/ncaa_net.csv")
print("NCAA NET shape:", ncaa.shape)

f538 = pd.read_csv("data/fivethirtyeight_forecasts.csv")
print("538 Forecasts shape:", f538.shape)

wn = pd.read_csv("data/warrennolan_net.csv")
print("WarrenNolan NET shape:", wn.shape)


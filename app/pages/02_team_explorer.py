"""
pages/02_team_explorer.py  —  Tab 2: Team Explorer
Deep-dive profile for any tournament team.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.style import inject_css, page_header
from utils.data_loader import load_teams, load_pairwise_probs, load_sim_results
from components.team_card import render_team_card
from components.matchup_card import render_matchup_card

st.set_page_config(page_title="Team Explorer | March Madness ML",
                   layout="wide", page_icon="🔍")
inject_css()
page_header("Team Explorer",
            "Pick any of the 68 tournament teams for a full statistical deep-dive")

# ── Load data ─────────────────────────────────────────────────────────────────
df_teams = load_teams()
df_probs = load_pairwise_probs()
df_sim   = load_sim_results()

all_teams = sorted(df_teams["team_name"].tolist())

# ── Sidebar: team picker ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Select Team")
    conf_filter = st.selectbox(
        "Filter by conference",
        ["All"] + sorted(df_teams["conf"].dropna().unique().tolist()),
    )
    filtered = df_teams if conf_filter == "All" else df_teams[df_teams["conf"] == conf_filter]
    team_options = sorted(filtered["team_name"].tolist())
    selected_team = st.selectbox("Team", team_options, index=0)

    st.markdown("---")
    st.markdown("### Head-to-Head")
    opp_options = [t for t in all_teams if t != selected_team]
    opponent = st.selectbox("Compare vs", ["(select)"] + opp_options, index=0)

# ── Main team card ────────────────────────────────────────────────────────────
render_team_card(selected_team, df_teams, df_sim)

st.markdown("---")

# ── Conference leaderboard ────────────────────────────────────────────────────
st.markdown("### Conference Rankings (by Champion %)")
conf_row = df_teams[df_teams["team_name"] == selected_team]
if len(conf_row):
    my_conf = conf_row.iloc[0]["conf"]
    conf_teams = df_sim[df_sim["conf"] == my_conf].copy().sort_values("rd6_pct", ascending=False)
    if len(conf_teams):
        fig = go.Figure(go.Bar(
            x=conf_teams["team_name"],
            y=conf_teams["rd6_pct"],
            marker_color=["#FFD700" if t == selected_team else "#1976D2"
                          for t in conf_teams["team_name"]],
            text=[f"{v:.1f}%" for v in conf_teams["rd6_pct"]],
            textposition="outside",
            textfont=dict(color="#e8ecf0", size=9),
        ))
        fig.update_layout(
            paper_bgcolor="#0d1b2a", plot_bgcolor="#1e2a3a",
            yaxis=dict(title="Champion %", gridcolor="#2e4060",
                       tickfont=dict(color="#9ab")),
            xaxis=dict(tickfont=dict(color="#e8ecf0")),
            margin=dict(l=20, r=20, t=20, b=60),
            height=280,
            title=dict(text=f"{my_conf} Conference — Champion Probabilities",
                       font=dict(color="#e8ecf0")),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Head-to-head matchup card ─────────────────────────────────────────────────
if opponent != "(select)":
    st.markdown("---")
    st.markdown(f"## Head-to-Head: {selected_team} vs {opponent}")
    render_matchup_card(selected_team, opponent, df_probs, df_teams)

# ── Bubble chart: efficiency vs championship prob ─────────────────────────────
st.markdown("---")
st.markdown("### All 68 Teams: Efficiency vs Championship Probability")
# df_sim already has conf/seed_num/region from load_sim_results()'s internal merge
sim_merged = df_sim.merge(df_teams[["team_name","adjoe","adjde"]],
                          on="team_name", how="left")
sim_merged = sim_merged.dropna(subset=["adjoe","adjde","rd6_pct"])

from utils.style import conf_color as _cc
bubble_colors = [_cc(c) for c in sim_merged["conf"].fillna("")]

fig_bubble = go.Figure(go.Scatter(
    x=sim_merged["adjoe"],
    y=sim_merged["adjde"],
    mode="markers+text",
    text=sim_merged["team_name"],
    textposition="top center",
    textfont=dict(size=7, color="#9ab"),
    marker=dict(
        size=sim_merged["rd6_pct"].clip(lower=0.2) * 4 + 8,
        color=bubble_colors,
        opacity=0.8,
        line=dict(color="#0d1b2a", width=1),
    ),
    hovertemplate=(
        "<b>%{text}</b><br>"
        "Adj OE: %{x:.1f}<br>"
        "Adj DE: %{y:.1f}<br>"
        "Champion %: %{customdata:.1f}%<extra></extra>"
    ),
    customdata=sim_merged["rd6_pct"],
))
fig_bubble.update_layout(
    paper_bgcolor="#0d1b2a", plot_bgcolor="#1e2a3a",
    xaxis=dict(title="Adj. Offensive Efficiency →", gridcolor="#2e4060",
               tickfont=dict(color="#9ab")),
    yaxis=dict(title="← Adj. Defensive Efficiency (lower = better)", gridcolor="#2e4060",
               tickfont=dict(color="#9ab"), autorange="reversed"),
    height=520,
    margin=dict(l=40, r=20, t=20, b=40),
)

# Highlight selected team
sel_row = sim_merged[sim_merged["team_name"] == selected_team]
if len(sel_row):
    sr = sel_row.iloc[0]
    fig_bubble.add_annotation(
        x=sr["adjoe"], y=sr["adjde"],
        text=f"◀ {selected_team}",
        showarrow=True, arrowhead=2,
        font=dict(color="#FFD700", size=11),
        arrowcolor="#FFD700",
    )

st.plotly_chart(fig_bubble, use_container_width=True, config={"displayModeBar": False})
st.caption("Bubble size = championship probability · Upper-right = best offensive + defensive efficiency")

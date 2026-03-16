"""
pages/01_bracket.py  —  Tab 1: Live Bracket
Interactive full 68-team bracket with probability overlays.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from utils.style import inject_css, page_header
from utils.data_loader import load_teams, load_pairwise_probs, load_sim_results
from components.bracket_viz import render_bracket
from components.team_card import render_team_card

st.set_page_config(page_title="Bracket | March Madness ML", layout="wide", page_icon="🏀")
inject_css()
page_header("2026 Tournament Bracket",
            "Chalk-pick bracket coloured by conference · hover for probabilities")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Bracket Options")
    show_team_card = st.checkbox("Show team profile on click", value=True)
    selected_team  = st.selectbox(
        "Highlight team path",
        ["(none)"] + sorted(load_teams()["team_name"].dropna().astype(str).tolist()),
        index=0,
    )

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading bracket data…"):
    df_teams  = load_teams()
    df_probs  = load_pairwise_probs()
    df_sim    = load_sim_results()

# ── Key metrics strip ─────────────────────────────────────────────────────────
top3 = df_sim.head(3)
cols = st.columns(3)
for i, (_, row) in enumerate(top3.iterrows()):
    with cols[i]:
        st.metric(
            label=f"#{i+1} Champion Favourite",
            value=row["team_name"],
            delta=f"{row['rd6_pct']:.1f}% prob",
        )

st.markdown("---")

# ── Bracket ───────────────────────────────────────────────────────────────────
render_bracket(df_teams, df_probs, df_sim)

st.markdown("---")

# ── Team profile card (sidebar selection) ─────────────────────────────────────
if show_team_card and selected_team != "(none)":
    st.markdown(f"## Team Profile: {selected_team}")
    render_team_card(selected_team, df_teams, df_sim)

# ── Legend ────────────────────────────────────────────────────────────────────
with st.expander("📖 How to read this bracket"):
    st.markdown("""
    - **Team names** are coloured by **conference**.
    - The bracket shows the **chalk-pick winner** (highest model probability) at each slot.
    - **Hover** any team name to see: win probability for this round, and overall championship %.
    - Use the sidebar to highlight a specific team's projected path.
    - Probabilities come from an ensemble of XGBoost (60%) + Logistic Regression (40%).
    """)

"""
pages/04_upsets.py  —  Tab 4: Upset Detector
Teams where the model sees significantly higher upset probability than seed history implies.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import plotly.graph_objects as go
import streamlit as st

from utils.style import inject_css, page_header
from utils.data_loader import load_upset_candidates, load_pairwise_probs, load_teams, load_sim_results
from components.matchup_card import render_matchup_card
from components.probability_bar import render_mini_bar

st.set_page_config(page_title="Upset Detector | March Madness ML",
                   layout="wide", page_icon="⚡")
inject_css()
page_header("Upset Detector",
            "Where the model diverges from seed expectations — your edge over field pickers")

# ── Load data ─────────────────────────────────────────────────────────────────
df_upsets = load_upset_candidates()
df_probs  = load_pairwise_probs()
df_teams  = load_teams()
df_sim    = load_sim_results()

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    min_edge = st.slider("Min upset edge (%)", 0, 30, 3, step=1)
    seed_filter = st.multiselect(
        "Underdog seed(s)",
        options=sorted(df_upsets["seed_underdog"].unique().tolist()),
        default=[],
    )

filtered = df_upsets[df_upsets["upset_edge"] * 100 >= min_edge]
if seed_filter:
    filtered = filtered[filtered["seed_underdog"].isin(seed_filter)]
filtered = filtered.reset_index(drop=True)

# ── Headline metrics ──────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Upset Candidates", len(filtered))
col2.metric("Biggest Edge", f"{filtered['upset_edge'].max()*100:.1f}%" if len(filtered) else "—",
            help="Largest gap between model and seed-implied upset probability")
col3.metric("Avg Model Upset %",
            f"{filtered['model_upset_prob'].mean()*100:.1f}%" if len(filtered) else "—")
col4.metric("Avg Seed Implied %",
            f"{filtered['seed_implied_prob'].mean()*100:.1f}%" if len(filtered) else "—")

st.markdown("---")

# ── Upset candidates table ────────────────────────────────────────────────────
st.markdown("### Top Upset Candidates")
st.caption("Ranked by model upset edge — how much more the model likes the underdog vs seed history")

if len(filtered) == 0:
    st.info("No upset candidates match the current filters.")
else:
    # Displayable table
    display_cols = ["underdog","seed_underdog","favorite","seed_favorite",
                    "model_upset_prob","seed_implied_prob","upset_edge","conf_underdog"]
    display = filtered[display_cols].copy()
    display.columns = ["Underdog","Und. Seed","Favourite","Fav. Seed",
                        "Model Upset %","Seed Implied %","Edge","Underdog Conf"]
    display["Model Upset %"]    = (display["Model Upset %"]   * 100).round(1)
    display["Seed Implied %"]   = (display["Seed Implied %"]  * 100).round(1)
    display["Edge"]             = (display["Edge"]            * 100).round(1)

    st.dataframe(
        display.style.background_gradient(subset=["Edge"], cmap="RdYlGn")
                     .format({"Model Upset %": "{:.1f}%",
                              "Seed Implied %": "{:.1f}%",
                              "Edge": "{:+.1f}%"}),
        use_container_width=True, hide_index=True,
    )

    # ── Scatter: seed-implied vs model upset prob ─────────────────────────────
    st.markdown("### Model vs Seed-Implied Upset Probability")
    fig = go.Figure()
    # Diagonal = perfect agreement
    max_v = filtered["model_upset_prob"].max() * 1.1
    fig.add_trace(go.Scatter(
        x=[0, max_v], y=[0, max_v],
        mode="lines", line=dict(color="#9ab", dash="dot", width=1),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=filtered["seed_implied_prob"],
        y=filtered["model_upset_prob"],
        mode="markers+text",
        text=filtered["underdog"],
        textposition="top center",
        textfont=dict(size=7.5, color="#9ab"),
        marker=dict(
            size=filtered["upset_edge"] * 200 + 8,
            color=filtered["upset_edge"],
            colorscale=[[0,"#1565C0"],[0.5,"#F57C00"],[1.0,"#E64A19"]],
            showscale=True,
            colorbar=dict(title="Edge", tickfont=dict(color="#9ab"), title_font=dict(color="#9ab")),
            opacity=0.85,
            line=dict(color="#0d1b2a", width=1),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Seed Implied: %{x:.1%}<br>"
            "Model: %{y:.1%}<br>"
            "Edge: %{marker.color:.1%}<extra></extra>"
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#0d1b2a", plot_bgcolor="#1e2a3a",
        xaxis=dict(title="Seed-Implied Upset Probability →", tickformat=".0%",
                   gridcolor="#2e4060", tickfont=dict(color="#9ab")),
        yaxis=dict(title="Model Upset Probability →", tickformat=".0%",
                   gridcolor="#2e4060", tickfont=dict(color="#9ab")),
        height=480, margin=dict(l=50, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Points above the diagonal = model is more bullish on upset than seed history · Bubble size = edge magnitude")

    st.markdown("---")

    # ── Matchup deep-dive ─────────────────────────────────────────────────────
    st.markdown("### Deep-Dive: Specific Matchup")
    matchup_options = [f"{r['underdog']} ({r['seed_underdog']}) vs {r['favorite']} ({r['seed_favorite']})"
                       for _, r in filtered.head(20).iterrows()]
    selected_mu = st.selectbox("Select upset matchup to analyze:", matchup_options)

    if selected_mu:
        idx = matchup_options.index(selected_mu)
        match_row = filtered.iloc[idx]
        render_matchup_card(match_row["underdog"], match_row["favorite"], df_probs, df_teams)

# ── Bar chart: which seeds produce most upsets ─────────────────────────────────
st.markdown("---")
st.markdown("### Upset Edge Distribution by Seed Matchup")
if len(df_upsets):
    seed_mu = df_upsets.groupby("seed_underdog")["upset_edge"].mean().reset_index()
    seed_mu.columns = ["seed_underdog","avg_edge"]
    seed_mu = seed_mu.sort_values("avg_edge", ascending=False)
    fig_seed = go.Figure(go.Bar(
        x=seed_mu["seed_underdog"].astype(str),
        y=(seed_mu["avg_edge"] * 100).round(1),
        marker_color="#E64A19",
        text=[(f"{v:.1f}%") for v in seed_mu["avg_edge"] * 100],
        textposition="outside", textfont=dict(color="#e8ecf0"),
    ))
    fig_seed.update_layout(
        paper_bgcolor="#0d1b2a", plot_bgcolor="#1e2a3a",
        xaxis=dict(title="Underdog Seed", tickfont=dict(color="#e8ecf0")),
        yaxis=dict(title="Avg Upset Edge (%)", gridcolor="#2e4060", tickfont=dict(color="#9ab")),
        height=320, margin=dict(l=40, r=20, t=20, b=40),
        title=dict(text="Average Model Upset Edge by Underdog Seed",
                   font=dict(color="#e8ecf0")),
    )
    st.plotly_chart(fig_seed, use_container_width=True, config={"displayModeBar": False})

with st.expander("📖 How upset edge is computed"):
    st.markdown("""
    **Upset edge** = (Model upset probability) − (Seed-implied upset probability)

    Seed-implied rates are empirical historical win rates:
    - 1 vs 16: 1% upset rate  
    - 5 vs 12: 35%  
    - 8 vs 9: 51%  
    
    A positive edge means the model thinks the underdog is meaningfully more dangerous
    than their seeding alone would suggest. This is your actionable bracket insight.
    """)

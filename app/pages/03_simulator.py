"""
pages/03_simulator.py  —  Tab 3: Monte Carlo Simulator
Interactive bracket simulation with adjustable n_sims.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.style import inject_css, page_header
from utils.data_loader import load_teams, load_pairwise_probs
from utils.sim_engine import run_simulation

st.set_page_config(page_title="Simulator | March Madness ML",
                   layout="wide", page_icon="🎲")
inject_css()
page_header("Monte Carlo Simulator",
            "Run bracket simulations in real-time and explore the probability landscape")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Simulation Settings")
    n_sims = st.select_slider(
        "Number of simulations",
        options=[1_000, 2_500, 5_000, 10_000, 25_000, 50_000],
        value=10_000,
    )
    rng_seed = st.number_input("Random seed", value=42, step=1)
    run_btn  = st.button("▶  Run Simulation", type="primary", use_container_width=True)
    st.caption(f"~{n_sims // 1000}k simulations of the full 64-team bracket")

# ── Load base data ────────────────────────────────────────────────────────────
df_teams = load_teams()
df_probs = load_pairwise_probs()

# ── Run or load simulation ────────────────────────────────────────────────────
if "sim_results" not in st.session_state or run_btn:
    with st.spinner(f"Running {n_sims:,} bracket simulations…"):
        df_sim = run_simulation(df_probs, df_teams, n_sims=n_sims, rng_seed=int(rng_seed))
    st.session_state["sim_results"] = df_sim
    st.session_state["sim_n"]       = n_sims
else:
    df_sim = st.session_state["sim_results"]
    n_sims = st.session_state.get("sim_n", n_sims)

st.success(f"Simulation complete — {n_sims:,} tournaments run.")

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_summary, tab_regions, tab_seed, tab_matrix = st.tabs(
    ["📊 Summary", "🗺 By Region", "🌱 By Seed", "🔢 Probability Matrix"]
)

# ── Tab 1: Summary ─────────────────────────────────────────────────────────────
with tab_summary:
    # Top 10 champion probabilities
    top10 = df_sim.head(10).copy()
    colors = [
        "#FFD700" if i == 0 else "#C0C0C0" if i == 1 else "#CD7F32" if i == 2
        else "#1976D2" for i in range(len(top10))
    ]
    fig_champ = go.Figure(go.Bar(
        x=top10["team_name"], y=top10["rd6_pct"],
        marker_color=colors,
        text=[f"{v:.1f}%" for v in top10["rd6_pct"]],
        textposition="outside",
        textfont=dict(color="#e8ecf0"),
        hovertemplate="%{x}<br>Champion: %{y:.2f}%<extra></extra>",
    ))
    fig_champ.update_layout(
        title=dict(text="Top 10 Championship Contenders", font=dict(color="#e8ecf0", size=14)),
        paper_bgcolor="#0d1b2a", plot_bgcolor="#1e2a3a",
        yaxis=dict(title="Champion Probability (%)", gridcolor="#2e4060",
                   tickfont=dict(color="#9ab")),
        xaxis=dict(tickfont=dict(color="#e8ecf0"), tickangle=-15),
        margin=dict(l=20, r=20, t=50, b=60),
        height=380,
    )
    st.plotly_chart(fig_champ, use_container_width=True, config={"displayModeBar": False})

    # Round-by-round reach probabilities heatmap
    st.markdown("### Round Reach Probability — All Teams")
    round_cols = ["rd1_pct","rd2_pct","rd3_pct","rd4_pct","rd5_pct","rd6_pct"]
    round_lbls = ["R64","R32","S16","E8","F4","Champ"]
    heatmap_df = df_sim.sort_values("rd6_pct", ascending=False).head(30)
    z = heatmap_df[round_cols].values
    fig_heat = go.Figure(go.Heatmap(
        z=z, x=round_lbls,
        y=heatmap_df["team_name"].tolist(),
        colorscale=[[0,"#0d1b2a"],[0.3,"#1565C0"],[0.6,"#388E3C"],[1.0,"#FFD700"]],
        text=[[f"{v:.1f}%" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=8),
        hovertemplate="Team: %{y}<br>%{x}: %{z:.1f}%<extra></extra>",
    ))
    fig_heat.update_layout(
        paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
        yaxis=dict(tickfont=dict(color="#e8ecf0", size=8), autorange="reversed"),
        xaxis=dict(tickfont=dict(color="#e8ecf0")),
        margin=dict(l=140, r=20, t=20, b=20),
        height=600,
    )
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

# ── Tab 2: By Region ──────────────────────────────────────────────────────────
with tab_regions:
    region_map = df_teams.set_index("team_name")["region_name"].to_dict()
    df_sim["region_name"] = df_sim["team_name"].map(region_map).fillna("Unknown")
    regions = sorted(df_sim["region_name"].dropna().unique().tolist())
    reg_cols = st.columns(min(2, len(regions)))

    for i, region in enumerate(regions):
        with reg_cols[i % 2]:
            sub = df_sim[df_sim["region_name"] == region].sort_values("rd6_pct", ascending=False)
            fig_r = go.Figure(go.Bar(
                y=sub["team_name"], x=sub["rd6_pct"],
                orientation="h",
                marker_color="#1976D2",
                text=[f"{v:.1f}%" for v in sub["rd6_pct"]],
                textposition="outside",
                textfont=dict(color="#e8ecf0", size=9),
            ))
            fig_r.update_layout(
                title=dict(text=f"{region} Region", font=dict(color="#e8ecf0")),
                paper_bgcolor="#0d1b2a", plot_bgcolor="#1e2a3a",
                xaxis=dict(title="Champion %", gridcolor="#2e4060", tickfont=dict(color="#9ab")),
                yaxis=dict(tickfont=dict(color="#e8ecf0", size=9), autorange="reversed"),
                height=380, margin=dict(l=140, r=40, t=40, b=20),
            )
            st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})

# ── Tab 3: By Seed ────────────────────────────────────────────────────────────
with tab_seed:
    seed_map = df_teams.set_index("team_name")["seed_num"].to_dict()
    df_sim["seed_num"] = df_sim["team_name"].map(seed_map).fillna(0).astype(int)
    seed_stats = (
        df_sim.groupby("seed_num")[["rd3_pct","rd4_pct","rd5_pct","rd6_pct"]]
        .mean().reset_index()
    )
    fig_seed = go.Figure()
    for col, label, color in [
        ("rd6_pct","Champion","#FFD700"),
        ("rd5_pct","Final Four","#E64A19"),
        ("rd4_pct","Elite 8","#F57C00"),
        ("rd3_pct","Sweet 16","#388E3C"),
    ]:
        fig_seed.add_trace(go.Scatter(
            x=seed_stats["seed_num"], y=seed_stats[col],
            mode="lines+markers", name=label,
            line=dict(color=color, width=2),
            marker=dict(size=7, color=color),
        ))
    fig_seed.update_layout(
        title=dict(text="Average Round Reach by Seed", font=dict(color="#e8ecf0")),
        paper_bgcolor="#0d1b2a", plot_bgcolor="#1e2a3a",
        xaxis=dict(title="Seed Number", tickvals=list(range(1,17)),
                   gridcolor="#2e4060", tickfont=dict(color="#9ab")),
        yaxis=dict(title="Average Probability (%)", gridcolor="#2e4060",
                   tickfont=dict(color="#9ab")),
        legend=dict(font=dict(color="#e8ecf0"), bgcolor="#1e2a3a"),
        height=400, margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_seed, use_container_width=True, config={"displayModeBar": False})

    # Seed win rate table
    st.markdown("### Seed Win Rates vs Historical Baseline")
    historical = {1:99, 2:94, 3:85, 4:79, 5:65, 6:62, 7:61, 8:49,
                  9:51, 10:39, 11:38, 12:35, 13:21, 14:15, 15:6, 16:1}
    tbl_rows = []
    for _, row in seed_stats[seed_stats["seed_num"] > 0].iterrows():
        s = int(row["seed_num"])
        r32 = float(row["rd3_pct"])   # Sweet 16 reach as proxy for R32 win
        hist = historical.get(s, 50)
        tbl_rows.append({"Seed": s, "Model R64→R32 %": round(r32,1),
                          "Historical R64→R32 %": hist,
                          "Edge": round(r32 - hist, 1)})
    tbl_df = pd.DataFrame(tbl_rows)
    st.dataframe(
        tbl_df.style.applymap(
            lambda v: "color: #43A047" if isinstance(v, float) and v > 0
                      else "color: #E53935" if isinstance(v, float) and v < 0 else "",
            subset=["Edge"]
        ),
        use_container_width=True, hide_index=True,
    )

# ── Tab 4: Probability Matrix ─────────────────────────────────────────────────
with tab_matrix:
    st.markdown("### Pairwise Win Probability Matrix (top 20 teams)")
    top20_names = df_sim.head(20)["team_name"].tolist()
    matrix_data = []
    for ta in top20_names:
        row = []
        for tb in top20_names:
            if ta == tb:
                row.append(0.5)
            else:
                from utils.data_loader import get_matchup
                m = get_matchup(ta, tb)
                row.append(float(m.get("ensemble_prob", 0.5)) if m else 0.5)
        matrix_data.append(row)

    z_matrix = np.array(matrix_data)
    text_matrix = [[f"{v*100:.0f}%" for v in row] for row in matrix_data]
    fig_mat = go.Figure(go.Heatmap(
        z=z_matrix, x=top20_names, y=top20_names,
        colorscale=[[0,"#E53935"],[0.5,"#1e2a3a"],[1.0,"#43A047"]],
        zmid=0.5, zmin=0.0, zmax=1.0,
        text=text_matrix, texttemplate="%{text}",
        textfont=dict(size=8),
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>P(row wins): %{z:.3f}<extra></extra>",
    ))
    fig_mat.update_layout(
        paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
        xaxis=dict(tickfont=dict(color="#e8ecf0", size=8), tickangle=-30),
        yaxis=dict(tickfont=dict(color="#e8ecf0", size=8), autorange="reversed"),
        height=680,
        margin=dict(l=160, r=20, t=20, b=120),
    )
    st.plotly_chart(fig_mat, use_container_width=True, config={"displayModeBar": False})
    st.caption("Row = team whose win probability is shown · Green = favoured · Red = underdog")

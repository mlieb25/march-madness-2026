"""
components/team_card.py
Rich team profile card: radar chart + path-to-glory bars + natural-language summary.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.style import seed_badge, conf_badge, ROUND_COLORS


RADAR_METRICS = ["adjoe_pct", "adjde_pct", "barthag_pct", "sos_pct", "wab_pct", "adjt_pct"]
RADAR_LABELS  = ["Off. Efficiency", "Def. Efficiency\n(inverted)", "Power Rating",
                  "Strength of Sched.", "Wins Above Bubble", "Tempo"]

ROUND_ORDER   = ["rd1_pct","rd2_pct","rd3_pct","rd4_pct","rd5_pct","rd6_pct"]
ROUND_LABELS  = ["R64","R32","S16","E8","F4","Champ"]
ROUND_COLORS_LIST = ["#1565C0","#1976D2","#388E3C","#F57C00","#E64A19","#FFD700"]


def _make_radar(team_row: pd.Series, all_teams_df: pd.DataFrame) -> go.Figure:
    # Team values (0–100)
    vals = [float(team_row.get(m, 50) or 50) for m in RADAR_METRICS]
    # Tournament-average values
    avg  = [float(all_teams_df[m].mean()) if m in all_teams_df.columns else 50.0
            for m in RADAR_METRICS]

    cats = RADAR_LABELS + [RADAR_LABELS[0]]  # close the polygon
    t_v  = vals + [vals[0]]
    a_v  = avg  + [avg[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=t_v, theta=cats, fill="toself",
        fillcolor="rgba(21,101,192,0.25)", line=dict(color="#1976D2", width=2),
        name=str(team_row.get("team_name", "Team")),
    ))
    fig.add_trace(go.Scatterpolar(
        r=a_v, theta=cats, fill="toself",
        fillcolor="rgba(150,160,170,0.08)", line=dict(color="#9ab", width=1.5, dash="dot"),
        name="Tourn. Avg.",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#1e2a3a",
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#9ab", size=8),
                            gridcolor="#2e4060"),
            angularaxis=dict(tickfont=dict(color="#e8ecf0", size=9), gridcolor="#2e4060"),
        ),
        paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
        legend=dict(font=dict(color="#9ab", size=9), bgcolor="#0d1b2a"),
        margin=dict(l=40, r=40, t=30, b=30),
        height=280,
    )
    return fig


def _make_path_chart(sim_row: pd.Series) -> go.Figure:
    vals   = [float(sim_row.get(r, 0) or 0) for r in ROUND_ORDER]
    fig = go.Figure(go.Bar(
        x=ROUND_LABELS, y=vals,
        marker_color=ROUND_COLORS_LIST,
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(color="#e8ecf0", size=10),
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0d1b2a", plot_bgcolor="#1e2a3a",
        yaxis=dict(range=[0, 110], gridcolor="#2e4060", tickfont=dict(color="#9ab"),
                   title=dict(text="Probability (%)", font=dict(color="#9ab"))),
        xaxis=dict(tickfont=dict(color="#e8ecf0")),
        margin=dict(l=30, r=20, t=20, b=30),
        height=240,
    )
    return fig


def _natural_language_summary(team_row: pd.Series, sim_row: pd.Series,
                               all_teams_df: pd.DataFrame) -> str:
    name         = str(team_row.get("team_name", "This team"))
    seed         = int(team_row.get("seed_num", 8) or 8)
    champ_pct    = float(sim_row.get("rd6_pct", 0) or 0)
    ff_pct       = float(sim_row.get("rd5_pct", 0) or 0)
    conf         = str(team_row.get("conf", "") or "")

    # Seed descriptor
    if seed <= 3:
        sd = "national contender"
    elif seed <= 6:
        sd = "tournament threat"
    elif seed <= 10:
        sd = "dangerous mid-seeded team"
    else:
        sd = "Cinderella candidate"

    # Efficiency descriptor (based on barthag percentile)
    b_pct = float(team_row.get("barthag_pct", 50) or 50)
    if b_pct >= 80:
        ed = "elite"
    elif b_pct >= 60:
        ed = "above-average"
    elif b_pct >= 40:
        ed = "solid"
    else:
        ed = "below-average"

    # Comparison vs seed expectation
    # Expected champion % by seed (rough historical rates)
    seed_exp = {1:18.0, 2:9.0, 3:4.5, 4:2.0, 5:0.8, 6:0.4}
    exp = seed_exp.get(seed, 0.15)
    if champ_pct > exp * 1.3:
        comparison = "exceeding what"
    elif champ_pct < exp * 0.7:
        comparison = "underperforming what"
    else:
        comparison = "in line with what"

    return (
        f"**{name}** is a **#{seed} seed** and a {sd} out of the {conf}. "
        f"The model rates their overall quality as **{ed}**. "
        f"They have a **{champ_pct:.1f}%** chance to win the title and "
        f"a **{ff_pct:.1f}%** shot at the Final Four — "
        f"{comparison} their seeding implies."
    )


def render_team_card(
    team_name: str,
    df_teams: pd.DataFrame,
    df_sim: pd.DataFrame,
) -> None:
    """Render a full team profile card (radar + path chart + key stats + NL summary)."""
    team_row = df_teams[df_teams["team_name"] == team_name]
    sim_row  = df_sim[df_sim["team_name"] == team_name]

    if len(team_row) == 0:
        st.warning(f"Team **{team_name}** not found in dataset.")
        return
    tr = team_row.iloc[0]
    sr = sim_row.iloc[0] if len(sim_row) > 0 else pd.Series(dtype=float)

    # ── Header ────────────────────────────────────────────────────────────────
    seed = int(tr.get("seed_num", 0) or 0)
    conf = str(tr.get("conf", "") or "")
    region = str(tr.get("region_name", "") or "")
    record = str(tr.get("record", "") or "")
    net    = int(tr.get("net_rank", 0) or 0)

    st.markdown(
        f"## {team_name}  "
        f"{seed_badge(seed)}  {conf_badge(conf)}",
        unsafe_allow_html=True,
    )
    st.caption(f"**Region:** {region}  ·  **Record:** {record}  ·  **NET:** #{net}")

    col_left, col_right = st.columns([1, 1.1], gap="large")

    # ── Left: Radar chart ─────────────────────────────────────────────────────
    with col_left:
        st.markdown('<div class="section-header">Efficiency Profile</div>', unsafe_allow_html=True)
        radar = _make_radar(tr, df_teams)
        st.plotly_chart(radar, use_container_width=True, config={"displayModeBar": False})

        # Key stats table
        adjoe = tr.get("adjoe", None)
        adjde = tr.get("adjde", None)
        barthag = tr.get("barthag", None)
        wab   = tr.get("wab", None)
        q1    = str(tr.get("quad1", "—") or "—")
        q2    = str(tr.get("quad2", "—") or "—")

        def fmt(v, dp=1): return f"{v:.{dp}f}" if pd.notna(v) else "—"

        stats_html = f"""
        <table style="width:100%;border-collapse:collapse;font-size:0.82rem;color:#e8ecf0;">
        <tr style="color:#9ab;border-bottom:1px solid #2e4060;">
          <td style="padding:5px 8px;">Adj. OE</td>
          <td style="padding:5px 8px;">Adj. DE</td>
          <td style="padding:5px 8px;">Barthag</td>
        </tr>
        <tr style="font-weight:600;border-bottom:1px solid #2e4060;">
          <td style="padding:5px 8px;color:#43A047;">{fmt(adjoe)}</td>
          <td style="padding:5px 8px;color:#43A047;">{fmt(adjde)}</td>
          <td style="padding:5px 8px;color:#FFD700;">{fmt(barthag,3)}</td>
        </tr>
        <tr style="color:#9ab;">
          <td style="padding:5px 8px;">WAB</td>
          <td style="padding:5px 8px;">Q1 Rec.</td>
          <td style="padding:5px 8px;">Q2 Rec.</td>
        </tr>
        <tr style="font-weight:600;">
          <td style="padding:5px 8px;">{fmt(wab)}</td>
          <td style="padding:5px 8px;">{q1}</td>
          <td style="padding:5px 8px;">{q2}</td>
        </tr>
        </table>
        """
        st.markdown(stats_html, unsafe_allow_html=True)

    # ── Right: Path to Glory ──────────────────────────────────────────────────
    with col_right:
        st.markdown('<div class="section-header">Path to Glory</div>', unsafe_allow_html=True)
        if len(sr) > 0:
            path_fig = _make_path_chart(sr)
            st.plotly_chart(path_fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Run simulation to see round-by-round probabilities.")

        # Natural language summary
        st.markdown("---")
        st.markdown('<div class="section-header">Model Says</div>', unsafe_allow_html=True)
        if len(sr) > 0:
            summary = _natural_language_summary(tr, sr, df_teams)
            st.markdown(summary)
        else:
            st.markdown(f"*{team_name}* — simulation data not yet available.")

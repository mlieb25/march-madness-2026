"""
components/matchup_card.py
Head-to-head matchup comparison card with SHAP explainer button.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.probability_bar import render_probability_bar
from utils.style import seed_badge, conf_badge


STAT_ROWS = [
    ("adjoe",    "Adj. Off. Efficiency",  True),   # higher is better
    ("adjde",    "Adj. Def. Efficiency",  False),  # lower is better
    ("barthag",  "Power Rating",          True),
    ("sos",      "Strength of Schedule",  True),
    ("wab",      "Wins Above Bubble",     True),
    ("adjt",     "Adjusted Tempo",        None),   # neutral
    ("net_rank", "NET Rank",              False),  # lower is better
]


def _fmt(v, dp=1):
    try:
        return f"{float(v):.{dp}f}"
    except (TypeError, ValueError):
        return "—"


def render_matchup_card(
    team_a: str,
    team_b: str,
    df_pairwise: pd.DataFrame,
    df_teams: pd.DataFrame,
) -> None:
    """Render a full head-to-head matchup comparison card."""
    from utils.data_loader import get_matchup

    matchup = get_matchup(team_a, team_b)
    row_a   = df_teams[df_teams["team_name"] == team_a]
    row_b   = df_teams[df_teams["team_name"] == team_b]

    if len(row_a) == 0 or len(row_b) == 0:
        st.warning("One or both teams not found in dataset.")
        return

    ra = row_a.iloc[0]
    rb = row_b.iloc[0]

    prob_a = float(matchup.get("ensemble_prob", 0.5)) if matchup else 0.5
    prob_lower = float(matchup.get("prob_lower", prob_a - 0.05)) if matchup else prob_a - 0.05
    prob_upper = float(matchup.get("prob_upper", prob_a + 0.05)) if matchup else prob_a + 0.05
    seed_a = int(ra.get("seed_num", 0) or 0)
    seed_b = int(rb.get("seed_num", 0) or 0)
    conf_a = str(ra.get("conf", "") or "")
    conf_b = str(rb.get("conf", "") or "")

    # ── Header ────────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        st.markdown(f"### {team_a}")
        st.markdown(
            f"{seed_badge(seed_a)}&nbsp;{conf_badge(conf_a)}",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div style='text-align:center;font-size:1.4rem;color:#9ab;font-weight:700;"
            "padding-top:10px;'>VS</div>", unsafe_allow_html=True
        )
    with c3:
        st.markdown(f"### {team_b}")
        st.markdown(
            f"{seed_badge(seed_b)}&nbsp;{conf_badge(conf_b)}",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Central probability bar ───────────────────────────────────────────────
    render_probability_bar(
        team_a, team_b, prob_a,
        prob_lower=prob_lower, prob_upper=prob_upper,
        height=48,
    )

    st.markdown("---")

    # ── Stat comparison table ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Statistical Comparison</div>',
                unsafe_allow_html=True)

    rows_html = ""
    for col, label, higher_is_better in STAT_ROWS:
        va = ra.get(col)
        vb = rb.get(col)
        try:
            fa, fb = float(va), float(vb)
        except (TypeError, ValueError):
            fa = fb = None

        if fa is None or fb is None:
            color_a = color_b = "#e8ecf0"
            arrow_a = arrow_b = ""
        elif higher_is_better is True:
            if fa > fb:
                color_a, color_b = "#43A047", "#E53935"
                arrow_a, arrow_b = "▲", ""
            elif fb > fa:
                color_a, color_b = "#E53935", "#43A047"
                arrow_a, arrow_b = "", "▲"
            else:
                color_a = color_b = "#e8ecf0"
                arrow_a = arrow_b = ""
        elif higher_is_better is False:
            if fa < fb:
                color_a, color_b = "#43A047", "#E53935"
                arrow_a, arrow_b = "▼", ""
            elif fb < fa:
                color_a, color_b = "#E53935", "#43A047"
                arrow_a, arrow_b = "", "▼"
            else:
                color_a = color_b = "#e8ecf0"
                arrow_a = arrow_b = ""
        else:
            color_a = color_b = "#e8ecf0"
            arrow_a = arrow_b = ""

        dp = 3 if col == "barthag" else (0 if col == "net_rank" else 1)
        val_a = _fmt(fa, dp) if fa is not None else "—"
        val_b = _fmt(fb, dp) if fb is not None else "—"

        rows_html += f"""
        <tr>
          <td style="text-align:right;padding:6px 12px;color:{color_a};font-weight:600;">
            {arrow_a} {val_a}
          </td>
          <td style="text-align:center;padding:6px 8px;color:#9ab;font-size:0.82rem;">
            {label}
          </td>
          <td style="text-align:left;padding:6px 12px;color:{color_b};font-weight:600;">
            {val_b} {arrow_b}
          </td>
        </tr>"""

    table_html = f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.88rem;">
      <thead>
        <tr style="border-bottom:1px solid #2e4060;">
          <th style="text-align:right;padding:6px 12px;color:#e8ecf0;">{team_a}</th>
          <th style="text-align:center;padding:6px 8px;color:#9ab;">Metric</th>
          <th style="text-align:left;padding:6px 12px;color:#e8ecf0;">{team_b}</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>"""
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("---")

    # ── SHAP explainer button ────────────────────────────────────────────────
    favored = team_a if prob_a >= 0.5 else team_b
    btn_label = f"🔍 Why does the model favor **{favored}**?"

    if st.button(btn_label, key=f"shap_{team_a}_{team_b}"):
        with st.spinner("Computing SHAP values…"):
            from utils.shap_utils import compute_shap_for_matchup
            result = compute_shap_for_matchup(team_a, team_b)

        if result.get("error"):
            st.error(result["error"])
        else:
            st.markdown(
                f"**Model prediction:** {result['predicted_prob']*100:.1f}% chance {team_a} wins",
            )

            col_pos, col_neg = st.columns(2)
            with col_pos:
                st.markdown(f"**Factors favouring {team_a}**")
                for label, sv, fv in result.get("top5_positive", []):
                    st.markdown(
                        f"<span class='edge-positive'>+{sv:.3f}</span> &nbsp; {label} = {fv:.3f}",
                        unsafe_allow_html=True,
                    )
            with col_neg:
                st.markdown(f"**Factors favouring {team_b}**")
                for label, sv, fv in result.get("top5_negative", []):
                    st.markdown(
                        f"<span class='edge-negative'>{sv:.3f}</span> &nbsp; {label} = {fv:.3f}",
                        unsafe_allow_html=True,
                    )

            if result.get("waterfall_fig") is not None:
                st.pyplot(result["waterfall_fig"], clear_figure=True)

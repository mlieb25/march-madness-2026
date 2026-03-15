"""
pages/05_explainer.py  —  Tab 5: Model Explainer
SHAP-based model interpretability for any matchup, plus feature importance overview.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.style import inject_css, page_header
from utils.data_loader import load_teams, load_pairwise_probs
from utils.shap_utils import compute_shap_for_matchup, FEATURE_LABELS

st.set_page_config(page_title="Model Explainer | March Madness ML",
                   layout="wide", page_icon="🧠")
inject_css()
page_header("Model Explainer",
            "SHAP-based feature importance — understand why the model favours each team")

# ── Load data ─────────────────────────────────────────────────────────────────
df_teams = load_teams()
df_probs = load_pairwise_probs()
all_teams = sorted(df_teams["team_name"].tolist())

# ── Sidebar: matchup selector ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Select Matchup")
    team_a = st.selectbox("Team A", all_teams, index=0)
    team_b_opts = [t for t in all_teams if t != team_a]
    team_b = st.selectbox("Team B", team_b_opts, index=0)
    explain_btn = st.button("🔍 Explain this matchup", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### About SHAP")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) decomposes each prediction into contributions
    from each feature. A positive SHAP value pushes the prediction toward Team A winning;
    negative pushes toward Team B.
    """)

# ── Global feature importance (static, from XGB trained in-memory) ────────────
st.markdown("## Global Feature Importance")
st.caption("Average |SHAP| across all 2026 matchups — which features drive model decisions most")

@st.cache_data(ttl=7200, show_spinner=False)
def _global_importance() -> pd.DataFrame:
    """Approximate global feature importance from XGBoost feature_importances_."""
    try:
        from utils.shap_utils import _get_trained_xgb, FEATURES
        clf = _get_trained_xgb()
        if clf is None:
            return pd.DataFrame()
        importances = clf.feature_importances_
        labels = [FEATURE_LABELS.get(f, f) for f in FEATURES]
        df = pd.DataFrame({"feature": labels, "importance": importances})
        return df.sort_values("importance", ascending=True)
    except Exception:
        return pd.DataFrame()

gi_df = _global_importance()
if len(gi_df):
    fig_gi = go.Figure(go.Bar(
        x=gi_df["importance"],
        y=gi_df["feature"],
        orientation="h",
        marker_color=[
            "#FFD700" if v > gi_df["importance"].quantile(0.75) else
            "#1976D2" if v > gi_df["importance"].median() else "#2e4060"
            for v in gi_df["importance"]
        ],
        text=[f"{v:.3f}" for v in gi_df["importance"]],
        textposition="outside",
        textfont=dict(color="#e8ecf0", size=9),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig_gi.update_layout(
        paper_bgcolor="#0d1b2a", plot_bgcolor="#1e2a3a",
        xaxis=dict(title="Feature Importance (XGBoost gain)", gridcolor="#2e4060",
                   tickfont=dict(color="#9ab")),
        yaxis=dict(tickfont=dict(color="#e8ecf0", size=9)),
        height=420, margin=dict(l=220, r=80, t=20, b=40),
    )
    st.plotly_chart(fig_gi, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("Train the XGBoost model to see global feature importance.")

# ── Feature definitions ────────────────────────────────────────────────────────
with st.expander("📖 Feature Definitions"):
    st.markdown("""
    | Feature | Description |
    |---|---|
    | **Adj. Off. Efficiency** | Points scored per 100 possessions, adjusted for opponent quality |
    | **Adj. Def. Efficiency** | Points allowed per 100 possessions, adjusted for opponent quality |
    | **Barthag** | Overall power rating — probability of beating an average D1 team |
    | **Strength of Schedule** | Quality of opponents faced |
    | **Wins Above Bubble** | Wins vs expected for a bubble-level team |
    | **Adjusted Tempo** | Estimated possessions per 40 minutes |

    All features are computed as **differences** (Team A − Team B) and **ratios** (A / B).
    The model predicts P(Team A wins) from Team A's perspective.
    """)

st.markdown("---")

# ── Per-matchup SHAP explanation ──────────────────────────────────────────────
st.markdown("## Matchup-Level SHAP Explanation")

if explain_btn or ("shap_result" not in st.session_state):
    if explain_btn:
        with st.spinner(f"Computing SHAP for {team_a} vs {team_b}…"):
            result = compute_shap_for_matchup(team_a, team_b)
        st.session_state["shap_result"] = result
        st.session_state["shap_a"] = team_a
        st.session_state["shap_b"] = team_b
    else:
        result = None
else:
    result = st.session_state.get("shap_result")
    team_a = st.session_state.get("shap_a", team_a)
    team_b = st.session_state.get("shap_b", team_b)

if result is None:
    st.info("Select two teams in the sidebar and click **Explain this matchup**.")
elif result.get("error"):
    st.error(result["error"])
    st.info("Make sure `shap` and `xgboost` are installed: `pip install shap xgboost`")
else:
    prob = result["predicted_prob"]
    favored = team_a if prob >= 0.5 else team_b
    underdog = team_b if prob >= 0.5 else team_a
    fav_prob = prob if prob >= 0.5 else 1 - prob

    # ── Summary banner ───────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#1e2a3a,#243040);
                    border:1px solid #2e4060;border-radius:12px;padding:16px 24px;
                    margin-bottom:16px;">
          <div style="font-size:1.3rem;font-weight:700;color:#FFD700;">
            {favored} favoured — {fav_prob*100:.1f}% probability
          </div>
          <div style="color:#9ab;font-size:0.85rem;margin-top:4px;">
            XGBoost model prediction · SHAP expected value: {result['expected_value']:.3f}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown(f"#### 🟢 Factors favouring **{team_a}**")
        for label, sv, fv in result.get("top5_positive", []):
            st.markdown(
                f"<span style='color:#43A047;font-weight:600;'>+{sv:.3f}</span> &nbsp; "
                f"**{label}** = `{fv:.3f}`",
                unsafe_allow_html=True,
            )

    with col_neg:
        st.markdown(f"#### 🔴 Factors favouring **{team_b}**")
        for label, sv, fv in result.get("top5_negative", []):
            st.markdown(
                f"<span style='color:#E53935;font-weight:600;'>{sv:.3f}</span> &nbsp; "
                f"**{label}** = `{fv:.3f}`",
                unsafe_allow_html=True,
            )

    # ── Waterfall chart ───────────────────────────────────────────────────────
    st.markdown("#### SHAP Waterfall Chart")
    if result.get("waterfall_fig") is not None:
        st.pyplot(result["waterfall_fig"], clear_figure=True)
    else:
        # Fallback: Plotly bar if matplotlib figure failed
        svs  = result["shap_values"]
        lbls = result["feature_labels"]
        order = np.argsort(np.abs(svs))[::-1][:10]
        fig_wf = go.Figure(go.Bar(
            x=[svs[i] for i in order],
            y=[lbls[i] for i in order],
            orientation="h",
            marker_color=["#43A047" if svs[i] > 0 else "#E53935" for i in order],
        ))
        fig_wf.add_vline(x=0, line_color="#9ab", line_width=1)
        fig_wf.update_layout(
            paper_bgcolor="#0d1b2a", plot_bgcolor="#1e2a3a",
            xaxis=dict(title=f"SHAP value (→ favours {team_a})", gridcolor="#2e4060",
                       tickfont=dict(color="#9ab")),
            yaxis=dict(tickfont=dict(color="#e8ecf0", size=9)),
            height=380, margin=dict(l=220, r=40, t=20, b=40),
        )
        st.plotly_chart(fig_wf, use_container_width=True, config={"displayModeBar": False})

    # ── Full feature table ────────────────────────────────────────────────────
    with st.expander("📋 Full Feature Values & SHAP Table"):
        rows = []
        for feat, lbl, sv, fv in zip(
            result["feature_names"], result["feature_labels"],
            result["shap_values"], result["feature_values"]
        ):
            rows.append({"Feature": lbl, "Value": round(fv, 4), "SHAP": round(sv, 4),
                          "Direction": f"→ {team_a}" if sv > 0 else f"→ {team_b}"})
        tbl = pd.DataFrame(rows).sort_values("SHAP", key=abs, ascending=False)
        st.dataframe(tbl, use_container_width=True, hide_index=True)

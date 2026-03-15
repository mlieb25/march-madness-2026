"""
app.py  —  Home Page  |  March Madness ML 2026
"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import pandas as pd
import streamlit as st
from utils.style import inject_css

st.set_page_config(
    page_title="March Madness ML 2026",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "March Madness 2026 ML prediction system."},
)

inject_css()

# ── Real data from phase outputs ──────────────────────────────────────────────
ROOT     = APP_DIR.parent
DATA_DIR = ROOT / "data"

@st.cache_data(ttl=3600)
def _load_home_data():
    # Champion probs from Phase 6 simulation
    sim_path = DATA_DIR / "phase6_team_round_probs.csv"
    sim_df   = pd.read_csv(sim_path) if sim_path.exists() else pd.DataFrame()

    # Best calibrated models from Phase 4
    import json
    p4_path = DATA_DIR / "phase4_best_combos.json"
    p4      = json.loads(p4_path.read_text()) if p4_path.exists() else []

    # Phase 5 ensemble weights
    p5_path = DATA_DIR / "phase5_ensemble_weights.json"
    p5      = json.loads(p5_path.read_text()) if p5_path.exists() else {}

    # Pool EV from Phase 6
    ev_path = DATA_DIR / "phase6_pool_ev.csv"
    ev_df   = pd.read_csv(ev_path) if ev_path.exists() else pd.DataFrame()

    return sim_df, p4, p5, ev_df

sim_df, p4_combos, p5_weights, ev_df = _load_home_data()

# Best log loss across all phases
best_ll   = min((c["log_loss"] for c in p4_combos), default=0.5039)
best_model = next((f"{c['family']} + {c['calibrator']}" for c in p4_combos
                   if c["log_loss"] == best_ll), "elastic_net + isotonic")
top_team  = sim_df.iloc[0] if len(sim_df) else None
n_sims    = 10_000

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@keyframes fadeSlide {
  from { opacity: 0; transform: translateY(-12px); }
  to   { opacity: 1; transform: translateY(0); }
}
.hero-wrapper {
  animation: fadeSlide 0.55s ease both;
  background: linear-gradient(135deg, #0a1628 0%, #0d1b2a 45%, #112240 100%);
  border: 1px solid #1e3a5f;
  border-radius: 20px;
  padding: 40px 48px 32px;
  margin-bottom: 28px;
  position: relative;
  overflow: hidden;
}
.hero-wrapper::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(ellipse 60% 80% at 75% 50%,
              rgba(25,118,210,0.08) 0%, transparent 70%);
  pointer-events: none;
}
.hero-eyebrow {
  font-size: 0.75rem; font-weight: 700; letter-spacing: 0.18em;
  color: #1976D2; text-transform: uppercase; margin-bottom: 10px;
}
.hero-title {
  font-size: clamp(2rem, 4vw, 2.9rem);
  font-weight: 900; line-height: 1.1; margin-bottom: 14px;
  background: linear-gradient(110deg, #e8ecf0 10%, #1976D2 55%, #FFD700 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub {
  color: #7a9ab5; font-size: 1.02rem; max-width: 620px;
  line-height: 1.65; margin-bottom: 24px;
}
.pill-row { display: flex; flex-wrap: wrap; gap: 8px; }
.pill {
  background: rgba(25,118,210,0.12); border: 1px solid rgba(25,118,210,0.35);
  color: #7ab4e8; font-size: 0.78rem; font-weight: 600;
  padding: 4px 12px; border-radius: 20px; letter-spacing: 0.04em;
}
.pill.gold { background: rgba(255,215,0,0.10); border-color: rgba(255,215,0,0.3); color: #c9a800; }
</style>
<div class="hero-wrapper">
  <div class="hero-eyebrow">NCAA Tournament · 2026 Edition</div>
  <div class="hero-title">March Madness ML 2026</div>
  <div class="hero-sub">
    A full machine-learning pipeline for predicting the NCAA Tournament —
    featuring Elastic Net, LightGBM, Gaussian Process, calibrated ensembles,
    10,000-run Monte Carlo simulation, and SHAP explainability.
  </div>
  <div class="pill-row">
    <span class="pill">XGBoost</span>
    <span class="pill">LightGBM</span>
    <span class="pill">Elastic Net</span>
    <span class="pill">Gaussian Process</span>
    <span class="pill">BMA Ensemble</span>
    <span class="pill">Isotonic Calibration</span>
    <span class="pill gold">10,000 Simulations</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Top-line metrics ──────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric("Best Log Loss", f"{best_ll:.4f}",
              delta="Phase 4 calibrated",
              help=f"Best model: {best_model}")

with m2:
    n_matchups = 2016
    st.metric("Matchups Scored", f"{n_matchups:,}",
              delta="All C(68,2) pairs",
              help="Every possible first-round matchup from top-68 NET teams")

with m3:
    st.metric("Simulations", f"{n_sims:,}",
              delta="Monte Carlo bracket runs",
              help="Full 64-team bracket simulated 10,000 times")

with m4:
    st.metric("Features", "12",
              delta="6 diffs + 6 ratios",
              help="adjOE, adjDE, Barthag, SoS, WAB, Tempo — as differences and ratios")

with m5:
    if top_team is not None:
        champ_col = "Championship" if "Championship" in top_team.index else "rd6_pct"
        champ_pct = float(top_team[champ_col]) * (100 if float(top_team[champ_col]) <= 1 else 1)
        st.metric(
            "Top Contender",
            str(top_team["team"]) if "team" in top_team.index else "—",
            delta=f"{champ_pct:.1f}% champion prob",
        )
    else:
        st.metric("Top Contender", "—")

st.markdown("---")

# ── Top Championship Contenders (real Phase 6 data) ───────────────────────────
if len(sim_df):
    st.markdown("## 🏆 Championship Contenders")
    st.caption("From 10,000 Monte Carlo bracket simulations — Phase 6 output")

    champ_col = "Championship"
    r32_col   = "Round of 32"
    ff_col    = "Final Four"

    top8 = sim_df.head(8).copy()
    cols = st.columns(4)

    MEDAL = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣"]
    CARD_COLORS = ["#7B6000", "#4a4a4a", "#6b3a1f", "#1a2a3a",
                   "#1a2a3a", "#1a2a3a", "#1a2a3a", "#1a2a3a"]
    BORDER_COLORS = ["#FFD700", "#C0C0C0", "#CD7F32", "#2e4060",
                     "#2e4060", "#2e4060", "#2e4060", "#2e4060"]

    for i, (_, row) in enumerate(top8.iterrows()):
        with cols[i % 4]:
            team   = str(row.get("team", "—"))
            champ  = float(row.get(champ_col, 0))
            ff     = float(row.get(ff_col, 0))
            r32    = float(row.get(r32_col, 0))
            champ_pct = champ * 100 if champ <= 1 else champ
            ff_pct    = ff    * 100 if ff    <= 1 else ff
            r32_pct   = r32   * 100 if r32   <= 1 else r32

            bg  = CARD_COLORS[i]
            bdr = BORDER_COLORS[i]
            st.markdown(f"""
            <div style="
              background: linear-gradient(135deg, {bg}, #1e2a3a);
              border: 1px solid {bdr};
              border-radius: 14px;
              padding: 16px 18px;
              margin-bottom: 12px;
              position: relative;
            ">
              <div style="position:absolute;top:10px;right:14px;font-size:1.4rem;">{MEDAL[i]}</div>
              <div style="font-size:0.7rem;color:#9ab;text-transform:uppercase;
                          letter-spacing:0.1em;margin-bottom:4px;">#{i+1} Contender</div>
              <div style="font-size:1.05rem;font-weight:700;color:#e8ecf0;
                          margin-bottom:10px;padding-right:28px;">{team}</div>
              <div style="display:flex;justify-content:space-between;
                          border-top:1px solid rgba(255,255,255,0.06);
                          padding-top:8px;margin-top:4px;">
                <div style="text-align:center;">
                  <div style="font-size:1.1rem;font-weight:800;
                              color:{'#FFD700' if i==0 else '#e8ecf0'};">{champ_pct:.1f}%</div>
                  <div style="font-size:0.65rem;color:#9ab;margin-top:2px;">CHAMPION</div>
                </div>
                <div style="text-align:center;">
                  <div style="font-size:1.1rem;font-weight:800;color:#E64A19;">{ff_pct:.1f}%</div>
                  <div style="font-size:0.65rem;color:#9ab;margin-top:2px;">FINAL 4</div>
                </div>
                <div style="text-align:center;">
                  <div style="font-size:1.1rem;font-weight:800;color:#1976D2;">{r32_pct:.1f}%</div>
                  <div style="font-size:0.65rem;color:#9ab;margin-top:2px;">R32</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# ── Model Performance Summary ─────────────────────────────────────────────────
st.markdown("## 📊 Pipeline Performance at a Glance")

# Phase comparison table
col_left, col_right = st.columns([1.2, 0.8])

with col_left:
    st.markdown("#### Model Benchmark (Holdout Log Loss · lower = better)")

    model_rows = [
        {"Phase": "2", "Model": "Logistic Regression (baseline)",  "Log Loss": "0.5112", "Brier": "0.1732", "Status": "✅ Baseline"},
        {"Phase": "3", "Model": "XGBoost (Bayesian opt)",          "Log Loss": "0.5373", "Brier": "0.1792", "Status": "✅ Phase 3"},
        {"Phase": "3", "Model": "LightGBM (Bayesian opt)",         "Log Loss": "0.5467", "Brier": "0.1821", "Status": "✅ Phase 3"},
        {"Phase": "3", "Model": "Gaussian Process",                "Log Loss": "0.5531", "Brier": "0.1857", "Status": "✅ Phase 3"},
        {"Phase": "4", "Model": "Elastic Net + Isotonic",          "Log Loss": "0.4806", "Brier": "0.1592", "Status": "🏆 Best"},
        {"Phase": "5", "Model": "BMA Ensemble (3 models)",         "Log Loss": "~0.483", "Brier": "~0.160", "Status": "🚀 Final"},
    ]
    bench_df = pd.DataFrame(model_rows)

    def style_bench(df):
        def highlight(row):
            if "Best" in str(row.get("Status", "")):
                return ["background-color: rgba(255,215,0,0.08); color: #FFD700"] * len(row)
            elif "Final" in str(row.get("Status", "")):
                return ["background-color: rgba(25,118,210,0.08)"] * len(row)
            return [""] * len(row)
        return df.style.apply(highlight, axis=1)

    st.dataframe(style_bench(bench_df), use_container_width=True, hide_index=True)

with col_right:
    st.markdown("#### Ensemble Blend (Phase 5 BMA weights)")

    if p5_weights:
        bma = p5_weights.get("bma_weights", {})
        models_w = [k.replace("_", " ").title() for k in bma.keys()]
        values_w = list(bma.values())

        import plotly.graph_objects as go
        fig_pie = go.Figure(go.Pie(
            labels=models_w, values=values_w,
            hole=0.5,
            marker=dict(colors=["#1976D2", "#388E3C", "#F57C00"],
                        line=dict(color="#0d1b2a", width=2)),
            textfont=dict(color="#e8ecf0", size=10),
            hovertemplate="%{label}<br>Weight: %{value:.3f}<extra></extra>",
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
            legend=dict(font=dict(color="#9ab", size=9), bgcolor="rgba(0,0,0,0)"),
            height=240, margin=dict(l=0, r=0, t=0, b=0),
            annotations=[dict(text="BMA<br>Weights", x=0.5, y=0.5,
                              font=dict(color="#e8ecf0", size=11),
                              showarrow=False)],
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

        final_blend = p5_weights.get("final_blend", {})
        kelly = p5_weights.get("kelly_final_bankroll", {})
        if kelly:
            best_kelly = max(kelly, key=lambda k: kelly[k])
            st.caption(f"Best Kelly bankroll: **{best_kelly.replace('_', ' ')}** → ${kelly[best_kelly]:.2f}")
    else:
        st.info("Phase 5 weights not found.")

st.markdown("---")

# ── App Navigation Cards ──────────────────────────────────────────────────────
st.markdown("## Navigate the App")

card_data = [
    ("🗓", "Bracket",       "01_Bracket",        "#1565C0",
     "Full 68-team interactive bracket with chalk-pick probabilities and conference colours."),
    ("🔍", "Team Explorer", "02_Team_Explorer",  "#388E3C",
     "Per-team efficiency radar, round-by-round odds, and head-to-head matchup comparison."),
    ("🎲", "Simulator",     "03_Simulator",      "#6A1B9A",
     "Run up to 50k Monte Carlo bracket sims and explore the probability distribution."),
    ("⚡", "Upset Detector","04_Upsets",         "#E64A19",
     "Where the model diverges from seed expectations — your edge over field pickers."),
    ("🧠", "Explainer",     "05_Explainer",      "#00695C",
     "SHAP-powered model interpretability for any head-to-head matchup."),
    ("⚗️", "Experiment Lab","06_Experiments",    "#1565C0",
     "Full model history: CV search, calibration, ensemble design, simulation outputs."),
]

cols = st.columns(3)
for i, (icon, title, page, accent, desc) in enumerate(card_data):
    with cols[i % 3]:
        st.markdown(f"""
        <a href="{page}" style="text-decoration:none;">
        <div style="
          background: linear-gradient(145deg, #1a2535, #1e2a3a);
          border: 1px solid rgba(255,255,255,0.06);
          border-left: 3px solid {accent};
          border-radius: 14px;
          padding: 20px 22px;
          margin-bottom: 14px;
          cursor: pointer;
          transition: all 0.2s ease;
          min-height: 120px;
        ">
          <div style="font-size:1.6rem;margin-bottom:8px;">{icon}</div>
          <div style="font-weight:700;color:#e8ecf0;font-size:1rem;margin-bottom:6px;">{title}</div>
          <div style="color:#6a8099;font-size:0.82rem;line-height:1.5;">{desc}</div>
        </div></a>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Pipeline Status ───────────────────────────────────────────────────────────
st.markdown("## 🔧 Pipeline Status")

phases = [
    ("Phase 1", "ETL & Feature Factory",          "data/ml_training_data.csv",           "472 rows · 12 features"),
    ("Phase 2", "Baseline Logistic Regression",   "data/phase2_results.csv",             "LL 0.5112 · Brier 0.1732"),
    ("Phase 3", "Bayesian Model Search",          "data/phase3_cv_results.csv",          "4 families · 35 trials each"),
    ("Phase 4", "Calibration Search",             "data/phase4_calibration_results.csv", "Best: EN+Isotonic LL 0.4806"),
    ("Phase 5", "BMA Ensemble + Kelly Criterion", "data/phase5_ensemble_weights.json",   "3-model blend · meta-stacker"),
    ("Phase 6", "Monte Carlo Simulation",         "data/phase6_team_round_probs.csv",    "10k sims · upset paths · pool EV"),
]

pcols = st.columns(3)
for i, (phase, name, fpath, summary) in enumerate(phases):
    exists = (ROOT / fpath).exists()
    icon   = "✅" if exists else "⏳"
    color  = "#43A047" if exists else "#F57C00"
    with pcols[i % 3]:
        st.markdown(f"""
        <div style="
          background: #1a2535;
          border: 1px solid rgba(255,255,255,0.05);
          border-radius: 10px;
          padding: 12px 16px;
          margin-bottom: 10px;
        ">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
            <span style="font-size:1rem;">{icon}</span>
            <span style="font-size:0.72rem;font-weight:700;color:{color};
                         text-transform:uppercase;letter-spacing:0.08em;">{phase}</span>
          </div>
          <div style="font-size:0.88rem;font-weight:600;color:#e8ecf0;margin-bottom:3px;">{name}</div>
          <div style="font-size:0.75rem;color:#5a7a95;">{summary}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Methodology Expander ──────────────────────────────────────────────────────
with st.expander("📐 Methodology Overview", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Phase 1 — Feature Factory**
        - Barttorvik 2008–2026 × FiveThirtyEight historical outcomes
        - 12 pairwise features: 6 diffs + 6 ratios
        - Strict symmetry (row-doubling) → 472 balanced rows

        **Phase 2 — Baseline**
        - Logistic Regression with StandardScaler
        - Time-aware split (train ≤2013, test 2014+)
        - Log loss **0.5112** · Brier **0.1732**

        **Phase 3 — Bayesian Search**
        - 35 Optuna trials each: XGBoost, LightGBM, Elastic Net, GP
        - Best per family stored in `phase3_top_models.json`
        """)
    with c2:
        st.markdown("""
        **Phase 4 — Calibration**
        - Platt, Isotonic, Beta, Venn-ABERS on all Phase 3 models
        - Best: **Elastic Net + Isotonic** → LL 0.4806

        **Phase 5 — Ensemble**
        - BMA weights, meta-stacker, risk-adaptive blending
        - Kelly criterion bankroll simulation across 3 models
        - Final: 40% BMA + 40% stack + 20% risk-adaptive

        **Phase 6 — Simulation**
        - 10,000 Monte Carlo bracket runs (NumPy vectorised)
        - Outputs: round-reach probs, upset paths, pool expected value
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#3a5060;font-size:0.76rem;
            margin-top:40px;padding-top:16px;border-top:1px solid #1a2a3a;">
  Built with Streamlit · XGBoost · LightGBM · scikit-learn · Plotly · SHAP<br>
  Data: Barttorvik · FiveThirtyEight · NCAA NET · Kaggle March ML Mania 2026
</div>
""", unsafe_allow_html=True)

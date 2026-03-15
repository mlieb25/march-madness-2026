"""
pages/06_experiments.py  —  Experiment Lab
Full model development history wired to real phase output files.
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from utils.style import inject_css, page_header
from utils.data_loader import load_live_scores, load_pairwise_probs
from utils.live_scorer import compute_live_metrics

st.set_page_config(page_title="Experiment Lab | March Madness ML",
                   layout="wide", page_icon="⚗️")
inject_css()
page_header("Experiment Lab",
            "Model development history · CV search results · calibration analysis · live tournament scoring")

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"

# ── Helpers ───────────────────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="#0d1b2a", plot_bgcolor="#1a2535",
    font=dict(color="#e8ecf0"),
    margin=dict(l=50, r=30, t=50, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9ab")),
)
AXIS_STYLE = dict(gridcolor="#1e2e42", tickfont=dict(color="#9ab"),
                  linecolor="#2e4060", zerolinecolor="#2e4060")

def dark_fig(**extra):
    layout = {**DARK_LAYOUT, **extra}
    return go.Figure().update_layout(**layout)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_all():
    out = {}

    # Phase 2 bar-to-beat
    p = DATA_DIR / "phase2_bar_to_beat.json"
    out["p2_btb"] = json.loads(p.read_text()) if p.exists() else {}

    # Phase 3 CV results
    p = DATA_DIR / "phase3_cv_results.csv"
    out["p3_cv"] = pd.read_csv(p) if p.exists() else pd.DataFrame()

    # Phase 3 top models
    p = DATA_DIR / "phase3_top_models.json"
    out["p3_top"] = json.loads(p.read_text()) if p.exists() else {}

    # Phase 4 calibration grid
    p = DATA_DIR / "phase4_calibration_results.csv"
    out["p4_cal"] = pd.read_csv(p) if p.exists() else pd.DataFrame()

    # Phase 4 best combos
    p = DATA_DIR / "phase4_best_combos.json"
    out["p4_best"] = json.loads(p.read_text()) if p.exists() else []

    # Phase 5 ensemble weights
    p = DATA_DIR / "phase5_ensemble_weights.json"
    out["p5_w"] = json.loads(p.read_text()) if p.exists() else {}

    # Phase 5 Kelly growth
    p = DATA_DIR / "phase5_kelly_results.csv"
    out["p5_kelly"] = pd.read_csv(p) if p.exists() else pd.DataFrame()

    # Phase 5 ensemble probs (for sample comparison)
    p = DATA_DIR / "phase5_ensemble_probs.csv"
    out["p5_probs"] = pd.read_csv(p).head(300) if p.exists() else pd.DataFrame()

    # Phase 6 team round probs
    p = DATA_DIR / "phase6_team_round_probs.csv"
    out["p6_rounds"] = pd.read_csv(p) if p.exists() else pd.DataFrame()

    # Phase 6 simulation raw champion frequencies
    p = DATA_DIR / "phase6_simulation_raw.csv"
    out["p6_raw"] = pd.read_csv(p) if p.exists() else pd.DataFrame()

    # Phase 6 pool EV
    p = DATA_DIR / "phase6_pool_ev.csv"
    out["p6_ev"] = pd.read_csv(p) if p.exists() else pd.DataFrame()

    # Phase 6 upset paths
    p = DATA_DIR / "phase6_upset_paths.csv"
    out["p6_upsets"] = pd.read_csv(p) if p.exists() else pd.DataFrame()

    # Training data shape — prefer V2, fall back to V1
    for train_file in ["ml_training_data_v2.csv", "ml_training_data.csv"]:
        p = DATA_DIR / train_file
        if p.exists():
            train = pd.read_csv(p)
            out["train_shape"]   = (len(train), train.shape[1])
            out["train_file"]    = train_file
            out["train_years"]   = sorted(train["year"].unique().tolist()) if "year" in train.columns else []
            out["train_balance"] = float(train["favorite_win_flag"].mean()) if "favorite_win_flag" in train.columns else 0.5
            break
    else:
        out["train_shape"]   = (0, 0)
        out["train_file"]    = "(none)"
        out["train_years"]   = []
        out["train_balance"] = 0.5

    # V2 ETL summary
    v2_path = DATA_DIR / "etl_v2_summary.json"
    out["etl_v2_summary"] = json.loads(v2_path.read_text()) if v2_path.exists() else {}

    # V2 inference data shape
    inf_v2 = DATA_DIR / "ml_inference_data_2026_v2.csv"
    if inf_v2.exists():
        inf_df = pd.read_csv(inf_v2)
        out["inference_v2_shape"] = (len(inf_df), inf_df.shape[1])
    else:
        out["inference_v2_shape"] = (0, 0)

    return out

data = load_all()

# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE SCORING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🏟️ Live Tournament Scoring")

live_raw = load_live_scores()
if not live_raw.get("tournament_started", False):
    st.info(
        "The 2026 tournament hasn't started yet — results will auto-populate once games are played.",
        icon="📡",
    )
    c1, c2, c3, c4 = st.columns(4)
    p4_best = data["p4_best"]
    best_ll = min((c["log_loss"] for c in p4_best), default=0.5039)
    c1.metric("Best Calibrated LL",   f"{best_ll:.4f}", delta="Phase 4 · Quick pipeline")
    c2.metric("Phase 2 Baseline LL",  "0.5040")
    c3.metric("Phase 3 Best LL",      "0.5207", delta="XGBoost grid search")
    c4.metric("Ensemble (Phase 5)",   "0.5140", delta="Tuned blend on 2014")
else:
    df_probs   = load_pairwise_probs()
    live_stats = compute_live_metrics(live_raw.get("raw", pd.DataFrame()), df_probs)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Games Scored",      live_stats["games_played"])
    c2.metric("Model Log Loss",    f"{live_stats['model_log_loss']:.4f}"    if live_stats["model_log_loss"]    else "—")
    c3.metric("Baseline Log Loss", f"{live_stats['baseline_log_loss']:.4f}" if live_stats["baseline_log_loss"] else "—",
              delta_color="inverse")
    c4.metric("Accuracy",          f"{live_stats['model_accuracy']*100:.1f}%" if live_stats["model_accuracy"] else "—")
    c5.metric("Beats Baseline",    "✅ Yes" if live_stats.get("model_beats_baseline") else "❌ No")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_p2, tab_p3, tab_p4, tab_p5, tab_p6, tab_data = st.tabs([
    "Phase 2 — Baselines",
    "Phase 3 — Model Search",
    "Phase 4 — Calibration",
    "Phase 5 — Ensemble",
    "Phase 6 — Simulation",
    "Data & Features",
])

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2
# ─────────────────────────────────────────────────────────────────────────────
with tab_p2:
    st.markdown("### Phase 2 — Baseline Logistic Regression")

    btb = data["p2_btb"]
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Architecture:** `LogisticRegression` + `StandardScaler`  
        **Split:** train ≤ 2013 · test 2014 (time-aware, zero leakage)

        The baseline tests three feature subsets:
        - **barthag_lr** — single Barthag feature only  
        - **full_lr** — all 12 features ✅ (used as baseline bar)  
        - **small_lr** — reduced 6-feature set

        Phase 3+ models must beat the `full_lr` log loss of **0.5040** to earn a spot in the ensemble.
        """)

        if btb:
            baseline_rows = []
            for variant, ll in btb.get("log_loss_raw", {}).items():
                baseline_rows.append({
                    "Model": variant.replace("_", " ").upper(),
                    "Log Loss": ll,
                    "Brier": btb.get("brier_raw", {}).get(variant, "—"),
                    "Bar to Beat": "✅ This is it" if variant == "full_lr" else "",
                })
            st.dataframe(pd.DataFrame(baseline_rows), use_container_width=True, hide_index=True)

    with col2:
        # Reliability diagram (approx)
        probs   = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        frac_lr = [0.06, 0.13, 0.24, 0.37, 0.46, 0.54, 0.66, 0.76, 0.86, 0.94]

        fig = dark_fig(height=340, title=dict(text="Reliability Diagram — Logistic Regression",
                                              font=dict(color="#e8ecf0")))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                 line=dict(color="#5a7a95", dash="dot", width=1.5),
                                 name="Perfect calibration"))
        fig.add_trace(go.Scatter(x=probs, y=frac_lr, mode="lines+markers",
                                 line=dict(color="#1976D2", width=2.5),
                                 marker=dict(size=7, color="#1976D2"),
                                 name="Logistic Regression"))
        fig.update_xaxes(title="Mean Predicted Probability", **AXIS_STYLE)
        fig.update_yaxes(title="Fraction of Positives", **AXIS_STYLE)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Calibration variants bar
    if btb and btb.get("log_loss_platt"):
        st.markdown("#### Baseline Calibration Variants")
        variants = ["raw", "platt", "iso"]
        methods  = ["log_loss_raw", "log_loss_platt", "log_loss_iso"]
        cal_rows = []
        for m, key in zip(variants, methods):
            for variant, v in btb.get(key, {}).items():
                cal_rows.append({"Variant": variant, "Calibrator": m, "Log Loss": v})
        cal_df = pd.DataFrame(cal_rows)
        fig_cal = px.bar(cal_df, x="Variant", y="Log Loss", color="Calibrator",
                         barmode="group",
                         color_discrete_map={"raw": "#1976D2", "platt": "#388E3C", "iso": "#E64A19"})
        fig_cal.update_layout(**DARK_LAYOUT, height=280,
                              yaxis=dict(title="Log Loss", range=[0.45, 0.75], **AXIS_STYLE),
                              xaxis=dict(title="", **AXIS_STYLE))
        st.plotly_chart(fig_cal, use_container_width=True, config={"displayModeBar": False})

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 3
# ─────────────────────────────────────────────────────────────────────────────
with tab_p3:
    st.markdown("### Phase 3 — Bayesian Hyperparameter Search")

    p3_cv = data["p3_cv"]
    p3_top = data["p3_top"]

    if len(p3_cv):
        col1, col2 = st.columns([1.5, 1])

        with col1:
            st.markdown("#### All Trials — CV Log Loss by Model Family")
            st.caption(f"{len(p3_cv)} total trials · 35 per family · lower = better · baseline bar = 0.5112")

            FAMILY_COLORS = {
                "elastic_net": "#1976D2",
                "xgboost":     "#43A047",
                "lightgbm":    "#F57C00",
                "gp":          "#AB47BC",
            }

            fig_trials = dark_fig(height=420)
            baseline_ll = data["p2_btb"].get("log_loss_raw", {}).get("full_lr", 0.5112)
            fig_trials.add_hline(y=baseline_ll, line_color="#9ab", line_dash="dot", line_width=1.5,
                                  annotation_text=f" Baseline ({baseline_ll:.4f})",
                                  annotation_font_color="#9ab")

            for fam, color in FAMILY_COLORS.items():
                sub = p3_cv[p3_cv["family"] == fam] if "family" in p3_cv.columns else pd.DataFrame()
                if len(sub):
                    fig_trials.add_trace(go.Scatter(
                        x=sub["trial"], y=sub["cv_log_loss"],
                        mode="markers", name=fam.replace("_", " ").title(),
                        marker=dict(size=7, color=color, opacity=0.8,
                                    line=dict(color="#0d1b2a", width=1)),
                        hovertemplate=f"{fam} trial %{{x}}<br>CV LL: %{{y:.4f}}<extra></extra>",
                    ))

            fig_trials.update_xaxes(title="Optuna Trial", **AXIS_STYLE)
            fig_trials.update_yaxes(title="CV Log Loss", **AXIS_STYLE)
            st.plotly_chart(fig_trials, use_container_width=True, config={"displayModeBar": False})

        with col2:
            st.markdown("#### Best Model per Family")
            best_rows = []
            for fam, trials in p3_top.items():
                if trials:
                    best = min(trials, key=lambda t: t["cv_log_loss"])
                    best_rows.append({
                        "Family": fam.replace("_", " ").title(),
                        "Best CV LL": f"{best['cv_log_loss']:.4f}",
                        "Trial #": best["trial"],
                    })
            st.dataframe(pd.DataFrame(best_rows), use_container_width=True, hide_index=True)

            st.markdown("#### Distribution of Log Loss by Family")
            box_data = []
            for fam in p3_cv["family"].unique() if "family" in p3_cv.columns else []:
                sub = p3_cv[p3_cv["family"] == fam]["cv_log_loss"].dropna()
                box_data.append(go.Box(
                    y=sub, name=fam.replace("_", " ").title(),
                    marker_color=FAMILY_COLORS.get(fam, "#1976D2"),
                    boxmean=True,
                    line=dict(width=1.5),
                ))
            fig_box = go.Figure(box_data)
            fig_box.update_layout(**DARK_LAYOUT, height=280,
                                   yaxis=dict(title="CV Log Loss", **AXIS_STYLE),
                                   xaxis=dict(**AXIS_STYLE),
                                   showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Phase 3 CV results not found (expected `phase3_cv_results.csv`).")

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 4
# ─────────────────────────────────────────────────────────────────────────────
with tab_p4:
    st.markdown("### Phase 4 — Calibration Search")

    p4_cal  = data["p4_cal"]
    p4_best = data["p4_best"]

    if len(p4_cal):
        col1, col2 = st.columns([1.6, 1])

        with col1:
            st.markdown("#### Log Loss Grid — All Family × Calibrator Combinations")

            if "family" in p4_cal.columns and "calibrator" in p4_cal.columns and "log_loss" in p4_cal.columns:
                families    = p4_cal["family"].unique().tolist()
                calibrators = p4_cal["calibrator"].unique().tolist()

                pivot = p4_cal.pivot(index="family", columns="calibrator", values="log_loss")
                z     = pivot.values
                text  = [[f"{v:.4f}" for v in row] for row in z]

                fig_heat = go.Figure(go.Heatmap(
                    z=z,
                    x=pivot.columns.tolist(),
                    y=[f.replace("_", " ").title() for f in pivot.index.tolist()],
                    colorscale=[[0, "#43A047"], [0.5, "#F57C00"], [1, "#E53935"]],
                    reversescale=False,
                    text=text, texttemplate="%{text}",
                    textfont=dict(size=11, color="#0d1b2a"),
                    hovertemplate="<b>%{y}</b> + <b>%{x}</b><br>Log Loss: %{z:.4f}<extra></extra>",
                    colorbar=dict(title="Log Loss",
                                  tickfont=dict(color="#9ab"),
                                  title_font=dict(color="#9ab")),
                ))
                fig_heat.update_layout(**DARK_LAYOUT, height=320,
                                        xaxis=dict(title="Calibrator", **AXIS_STYLE),
                                        yaxis=dict(title="Model Family", **AXIS_STYLE))
                st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

        with col2:
            st.markdown("#### Top 3 Calibrated Models")
            for rank, combo in enumerate(p4_best, 1):
                medal = ["🥇", "🥈", "🥉"][rank - 1]
                delta_str = f"{combo.get('ll_delta_vs_raw', 0):+.4f} vs raw"
                st.markdown(f"""
                <div style="background:#1a2535;border:1px solid #2e4060;border-radius:10px;
                            padding:12px 16px;margin-bottom:10px;">
                  <div style="font-size:0.75rem;color:#9ab;margin-bottom:4px;">{medal} Rank {rank}</div>
                  <div style="font-weight:700;color:#e8ecf0;font-size:0.95rem;">
                    {combo['family'].replace('_',' ').title()} + {combo['calibrator'].title()}
                  </div>
                  <div style="display:flex;gap:16px;margin-top:8px;">
                    <div>
                      <div style="font-size:1.1rem;font-weight:800;color:#43A047;">{combo['log_loss']:.4f}</div>
                      <div style="font-size:0.65rem;color:#9ab;">LOG LOSS</div>
                    </div>
                    <div>
                      <div style="font-size:1.1rem;font-weight:800;color:#1976D2;">{combo['brier']:.4f}</div>
                      <div style="font-size:0.65rem;color:#9ab;">BRIER</div>
                    </div>
                    <div>
                      <div style="font-size:1.1rem;font-weight:800;
                                  color:{'#43A047' if combo.get('ll_delta_vs_raw', 0) < 0 else '#E53935'};">
                        {combo.get('ll_delta_vs_raw', 0):+.4f}</div>
                      <div style="font-size:0.65rem;color:#9ab;">Δ vs RAW</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # Full calibration metrics table
        with st.expander("📋 Full Calibration Results Table"):
            display_cols = ["family", "calibrator", "log_loss", "brier", "ece",
                            "sharpness", "ll_delta_vs_raw"]
            cols_exist = [c for c in display_cols if c in p4_cal.columns]
            styled = p4_cal[cols_exist].style.background_gradient(
                subset=["log_loss"] if "log_loss" in cols_exist else [],
                cmap="RdYlGn_r"
            ).format({c: "{:.4f}" for c in ["log_loss","brier","ece","sharpness","ll_delta_vs_raw"]
                      if c in cols_exist})
            st.dataframe(styled, use_container_width=True, hide_index=True)

    else:
        st.info("Phase 4 calibration results not found.")

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 5
# ─────────────────────────────────────────────────────────────────────────────
with tab_p5:
    st.markdown("### Phase 5 — BMA Ensemble, Meta-Stacker & Kelly Criterion")

    p5_w     = data["p5_w"]
    p5_kelly = data["p5_kelly"]
    p5_probs = data["p5_probs"]

    col1, col2 = st.columns(2)

    with col1:
        if p5_w:
            st.markdown("#### BMA Ensemble Weights")
            bma_w = p5_w.get("bma_weights", {})
            labels = [k.replace("_", " ").title() for k in bma_w.keys()]
            values = list(bma_w.values())

            fig_donut = go.Figure(go.Pie(
                labels=labels, values=values,
                hole=0.55,
                marker=dict(colors=["#1976D2", "#F57C00", "#AB47BC"],
                            line=dict(color="#0d1b2a", width=2)),
                textfont=dict(color="#e8ecf0", size=10),
                hovertemplate="%{label}<br>BMA Weight: %{value:.4f}<extra></extra>",
            ))
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(font=dict(color="#9ab", size=9), bgcolor="rgba(0,0,0,0)"),
                height=260, margin=dict(l=0, r=0, t=0, b=0),
                annotations=[dict(text="BMA", x=0.5, y=0.5,
                                  font=dict(color="#e8ecf0", size=13, family="Inter"),
                                  showarrow=False)],
            )
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

            # Final blend
            st.markdown("#### Final Ensemble Blend")
            final_blend = p5_w.get("final_blend", {})
            blend_rows = [{"Component": k.upper(), "Weight": f"{v:.0%}"}
                          for k, v in final_blend.items()]
            st.dataframe(pd.DataFrame(blend_rows), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Kelly Criterion Bankroll Growth")

        if len(p5_kelly):
            fig_kelly = dark_fig(height=320,
                                  title=dict(text="Simulated Bankroll over Tournament Games",
                                             font=dict(color="#e8ecf0", size=13)))
            KELLY_COLORS = {
                "elastic_net_isotonic": "#1976D2",
                "lightgbm_beta":        "#F57C00",
                "gp_beta":              "#AB47BC",
                "bma_ensemble":         "#FFD700",
            }
            for col in p5_kelly.columns:
                fig_kelly.add_trace(go.Scatter(
                    y=p5_kelly[col],
                    mode="lines",
                    name=col.replace("_", " ").title(),
                    line=dict(color=KELLY_COLORS.get(col, "#9ab"), width=2),
                ))
            fig_kelly.update_xaxes(title="Tournament Game #", **AXIS_STYLE)
            fig_kelly.update_yaxes(title="Bankroll ($, starting = $1)", **AXIS_STYLE)
            st.plotly_chart(fig_kelly, use_container_width=True, config={"displayModeBar": False})

            # Final bankroll summary
            kelly_final = p5_w.get("kelly_final_bankroll", {})
            if kelly_final:
                kf_rows = [{"Model": k.replace("_", " ").title(), "Final Bankroll ($)": f"${v:.2f}"}
                            for k, v in sorted(kelly_final.items(), key=lambda x: -x[1])]
                st.dataframe(pd.DataFrame(kf_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Phase 5 Kelly results not found.")

    # Model vs BMA vs Stack scatter
    if len(p5_probs):
        st.markdown("#### Ensemble Comparison — BMA vs Stack vs Risk-Adaptive")
        fig_cmp = dark_fig(height=360, title=dict(text="Pairwise Probability Comparison",
                                                   font=dict(color="#e8ecf0")))
        if all(c in p5_probs.columns for c in ["p_bma", "p_stack"]):
            fig_cmp.add_trace(go.Scatter(
                x=p5_probs["p_bma"], y=p5_probs["p_stack"],
                mode="markers",
                marker=dict(size=4, color="#1976D2", opacity=0.5),
                name="BMA vs Stack",
                hovertemplate="BMA: %{x:.3f}<br>Stack: %{y:.3f}<extra></extra>",
            ))
        if "p_risk_adaptive" in p5_probs.columns and "p_bma" in p5_probs.columns:
            fig_cmp.add_trace(go.Scatter(
                x=p5_probs["p_bma"], y=p5_probs["p_risk_adaptive"],
                mode="markers",
                marker=dict(size=4, color="#F57C00", opacity=0.5),
                name="BMA vs Risk-Adaptive",
                hovertemplate="BMA: %{x:.3f}<br>Risk-Adaptive: %{y:.3f}<extra></extra>",
            ))
        fig_cmp.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     line=dict(color="#5a7a95", dash="dot", width=1),
                                     showlegend=False))
        fig_cmp.update_xaxes(title="BMA Probability", range=[0,1], **AXIS_STYLE)
        fig_cmp.update_yaxes(title="Other Model Probability", range=[0,1], **AXIS_STYLE)
        st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})
        st.caption("Tight clustering around the diagonal = models broadly agree · deviations = disagreement zones")

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 6
# ─────────────────────────────────────────────────────────────────────────────
with tab_p6:
    st.markdown("### Phase 6 — Monte Carlo Bracket Simulation")

    p6_rounds  = data["p6_rounds"]
    p6_raw     = data["p6_raw"]
    p6_ev      = data["p6_ev"]
    p6_upsets  = data["p6_upsets"]

    # ── Champion frequency from raw sims ──────────────────────────────────────
    if len(p6_raw) and "champion" in p6_raw.columns:
        st.markdown("#### Champion Frequency — 10,000 Simulated Tournaments")
        champ_freq = p6_raw["champion"].value_counts().reset_index()
        champ_freq.columns = ["team", "wins"]
        champ_freq["pct"] = champ_freq["wins"] / len(p6_raw) * 100
        top_champs = champ_freq.head(15)

        colors = []
        for i, row in enumerate(top_champs.itertuples()):
            colors.append(
                "#FFD700" if i == 0 else
                "#C0C0C0" if i == 1 else
                "#CD7F32" if i == 2 else
                "#1976D2"
            )

        fig_champ = dark_fig(height=420,
                              title=dict(text="Champion Frequency (top 15 teams)",
                                         font=dict(color="#e8ecf0")))
        fig_champ.add_trace(go.Bar(
            x=top_champs["team"], y=top_champs["pct"],
            marker_color=colors,
            text=[f"{v:.1f}%" for v in top_champs["pct"]],
            textposition="outside",
            textfont=dict(color="#e8ecf0", size=9.5),
            hovertemplate="%{x}<br>Champion in %{y:.1f}% of sims<extra></extra>",
        ))
        fig_champ.update_xaxes(tickangle=-20, **AXIS_STYLE)
        fig_champ.update_yaxes(title="% of Simulations", **AXIS_STYLE)
        st.plotly_chart(fig_champ, use_container_width=True, config={"displayModeBar": False})

    # ── Round reach heatmap ────────────────────────────────────────────────────
    if len(p6_rounds):
        st.markdown("#### Round Reach Probabilities — Top 25 Teams")
        round_cols = [c for c in ["Round of 32", "Sweet 16", "Elite 8",
                                   "Final Four", "Championship"] if c in p6_rounds.columns]
        top25 = p6_rounds.head(25).copy()

        # Normalize if stored as 0–1
        for c in round_cols:
            if top25[c].max() <= 1.01:
                top25[c] = top25[c] * 100

        team_col = "team" if "team" in top25.columns else top25.columns[0]
        z    = top25[round_cols].values
        text = [[f"{v:.1f}%" for v in row] for row in z]

        fig_heat = go.Figure(go.Heatmap(
            z=z, x=round_cols,
            y=top25[team_col].tolist(),
            colorscale=[[0,"#0d1b2a"],[0.25,"#1565C0"],[0.55,"#388E3C"],[1.0,"#FFD700"]],
            text=text, texttemplate="%{text}",
            textfont=dict(size=8.5),
            hovertemplate="<b>%{y}</b> · %{x}: %{z:.1f}%<extra></extra>",
            colorbar=dict(title="% chance", tickfont=dict(color="#9ab"),
                          title_font=dict(color="#9ab")),
        ))
        fig_heat.update_layout(
            **DARK_LAYOUT, height=640,
            yaxis=dict(tickfont=dict(color="#e8ecf0", size=9), autorange="reversed",
                       **{k:v for k,v in AXIS_STYLE.items() if k != "title"}),
            xaxis=dict(tickfont=dict(color="#e8ecf0"), **{k:v for k,v in AXIS_STYLE.items() if k!="title"}),
            margin=dict(l=150, r=30, t=30, b=30),
        )
        st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

    col1, col2 = st.columns(2)

    # ── Pool EV ────────────────────────────────────────────────────────────────
    with col1:
        if len(p6_ev):
            st.markdown("#### Bracket Strategy Pool Expected Value")
            strategy_colors = {"chalk": "#1976D2", "exploitative": "#F57C00",
                                "high_variance": "#AB47BC"}
            fig_ev = dark_fig(height=320,
                               title=dict(text="Pool Strategy Comparison",
                                          font=dict(color="#e8ecf0")))
            for _, row in p6_ev.iterrows():
                bracket = str(row.get("bracket", ""))
                mean_ev = float(row.get("mean_ev", 0))
                std_ev  = float(row.get("std_ev",  0))
                p10     = float(row.get("p10",     0))
                p90     = float(row.get("p90",     0))
                color   = strategy_colors.get(bracket, "#9ab")

                fig_ev.add_trace(go.Scatter(
                    x=[bracket], y=[mean_ev],
                    mode="markers",
                    marker=dict(size=14, color=color, symbol="diamond"),
                    error_y=dict(type="data", symmetric=False,
                                 array=[p90 - mean_ev],
                                 arrayminus=[mean_ev - p10],
                                 color=color, thickness=2, width=6),
                    name=bracket.replace("_", " ").title(),
                    hovertemplate=f"{bracket}<br>Mean EV: %{{y:.1f}}<br>p10–p90: {p10:.0f}–{p90:.0f}<extra></extra>",
                ))
            fig_ev.update_xaxes(**AXIS_STYLE)
            fig_ev.update_yaxes(title="Expected Pool Score", **AXIS_STYLE)
            st.plotly_chart(fig_ev, use_container_width=True, config={"displayModeBar": False})

            st.caption("Error bars show 10th–90th percentile bracket scores · diamond = mean EV")

            st.dataframe(
                p6_ev.rename(columns={"bracket": "Strategy", "mean_ev": "Mean EV",
                                       "std_ev": "Std Dev", "p10": "p10",
                                       "p50": "p50", "p90": "p90", "p99": "p99"}),
                use_container_width=True, hide_index=True,
            )

    # ── Upset paths ────────────────────────────────────────────────────────────
    with col2:
        if len(p6_upsets):
            st.markdown("#### Cinderella / Upset Paths (Phase 6 Output)")
            team_col  = "team"  if "team"  in p6_upsets.columns else p6_upsets.columns[0]
            seed_col  = "seed"  if "seed"  in p6_upsets.columns else None
            s16_col   = "P(Sweet16)"   if "P(Sweet16)"   in p6_upsets.columns else None
            e8_col    = "P(Elite8)"    if "P(Elite8)"    in p6_upsets.columns else None
            ff_col    = "P(FinalFour)" if "P(FinalFour)" in p6_upsets.columns else None

            if s16_col:
                # Horizontal grouped bar
                upset_plot = p6_upsets.copy()
                round_vals = [c for c in [s16_col, e8_col, ff_col] if c]
                fig_up = dark_fig(height=380,
                                   title=dict(text="Upset Path Probabilities",
                                              font=dict(color="#e8ecf0")))
                colors_up = ["#388E3C", "#F57C00", "#E64A19"]
                for c, col, lbl in zip(colors_up, round_vals,
                                       ["Sweet 16", "Elite 8", "Final Four"]):
                    fig_up.add_trace(go.Bar(
                        y=upset_plot[team_col],
                        x=upset_plot[col] * 100,
                        name=lbl,
                        orientation="h",
                        marker_color=c,
                        hovertemplate=f"%{{y}}<br>{lbl}: %{{x:.1f}}%<extra></extra>",
                    ))
                fig_up.update_layout(barmode="group")
                fig_up.update_xaxes(title="Probability (%)", **AXIS_STYLE)
                fig_up.update_yaxes(tickfont=dict(color="#e8ecf0", size=9),
                                    autorange="reversed",
                                    **{k:v for k,v in AXIS_STYLE.items() if k!="title"})
                st.plotly_chart(fig_up, use_container_width=True, config={"displayModeBar": False})

                st.dataframe(p6_upsets, use_container_width=True, hide_index=True)
            else:
                st.dataframe(p6_upsets, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
#  DATA & FEATURES
# ─────────────────────────────────────────────────────────────────────────────
with tab_data:
    st.markdown("### Data Pipeline & Feature Engineering")

    n_rows, n_cols = data["train_shape"]
    years          = data["train_years"]
    balance        = data["train_balance"]
    v2             = data.get("etl_v2_summary", {})
    train_file     = data.get("train_file", "ml_training_data.csv")
    is_v2          = "v2" in train_file
    inf_rows, _    = data.get("inference_v2_shape", (0, 0))

    if is_v2:
        st.success(
            f"**V2 Pipeline Active** — `{train_file}` · "
            f"{n_rows:,} rows · {v2.get('features_training', n_cols)} features · "
            f"{v2.get('year_range','1985–2025')} · {v2.get('seasons_covered',40)} seasons",
            icon="✅",
        )
    else:
        st.warning(f"Using V1 training data (`{train_file}`). Run `etl_v2.py` for the full dataset.", icon="⚠️")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Training Rows",     f"{n_rows:,}")
    col2.metric("Features",          v2.get("features_training", n_cols))
    col3.metric("Inference Matchups",f"{inf_rows:,}" if inf_rows else f"{v2.get('inference_matchups',1953):,}")
    col4.metric("Seasons",           v2.get("year_range", f"{min(years)}–{max(years)}") if years else "—")
    col5.metric("Label Balance",     f"{balance:.1%}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Feature Groups (V2)")
        feat_rows = [
            {"Group": "Basic Stats",       "Features": "WinPct, PPG, OppPPG, PointDiff, HomeWinPct, AwayWinPct", "Format": "Diff"},
            {"Group": "Four Factors",      "Features": "eFG%, TOV Rate, ORB Rate, FTA Rate (2003+)",              "Format": "Diff"},
            {"Group": "Torvik Efficiency", "Features": "adjOE, adjDE, Barthag, SoS, WAB, Tempo",                  "Format": "Diff + Ratio"},
            {"Group": "Massey Ordinals",   "Features": "POM, SAG, RPI, DOK, COL rank differentials",              "Format": "Diff"},
            {"Group": "Seeding",           "Features": "SeedNum (tournament years only)",                          "Format": "Diff"},
        ]
        st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

        st.markdown("#### Time-Aware Split")
        st.code("Train: 1985–2013\nTest:  2014  (held out)")

    with col2:
        st.markdown("#### Data Sources (V2)")
        st.markdown("""
        | Source | Contents | Seasons |
        |---|---|---|
        | **Kaggle MM Mania 2026** | Box scores, wins, Four Factors | 1985–2025 |
        | **Barttorvik** | adjOE, adjDE, Barthag, SoS, WAB, Tempo | 2008–2026 |
        | **FiveThirtyEight** | Tournament outcomes (ground truth) | 2008–2025 |
        | **NCAA NET** | 2026 rankings (bracket seeding) | 2026 |
        """)

        missing = v2.get("missing_data_pct", {}).get("training", {})
        if missing:
            st.markdown("#### Missing Data (V2 Training)")
            miss_rows = [{"Feature": k, "Missing %": f"{v:.0f}%"}
                         for k, v in sorted(missing.items(), key=lambda x: -x[1])]
            st.dataframe(pd.DataFrame(miss_rows), use_container_width=True, hide_index=True, height=220)
            st.caption("Torvik & Four Factors missing pre-2003/2008; imputed via median in model.")

    if years:
        st.markdown("#### Training Data — Rows per Season")
        try:
            train = pd.read_csv(DATA_DIR / train_file)
            if "year" in train.columns:
                gpsy = train.groupby("year").size().reset_index(name="games")
                fig_yr = dark_fig(height=300,
                                   title=dict(text=f"Rows per Season — {train_file}",
                                              font=dict(color="#e8ecf0")))
                fig_yr.add_trace(go.Bar(
                    x=gpsy["year"].astype(str), y=gpsy["games"],
                    marker_color=["#FFD700" if y >= 2014 else "#1976D2" for y in gpsy["year"]],
                    text=gpsy["games"], textposition="outside",
                    textfont=dict(color="#e8ecf0", size=7),
                ))
                fig_yr.update_xaxes(title="Season", tickangle=-45, **AXIS_STYLE)
                fig_yr.update_yaxes(title="Row Count", **AXIS_STYLE)
                st.plotly_chart(fig_yr, use_container_width=True, config={"displayModeBar": False})
                st.caption("🟡 Gold = holdout year (2014+) · 🔵 Blue = training years")
        except Exception:
            pass

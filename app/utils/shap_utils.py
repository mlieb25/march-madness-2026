"""
utils/shap_utils.py
On-demand SHAP computation for individual matchups.
Retrains XGBoost in-memory (no pkl files required).
"""
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from pathlib import Path
_ROOT    = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _ROOT / "data"

FEATURES = [
    "adjoe_diff", "adjde_diff", "barthag_diff", "sos_diff", "wab_diff", "adjt_diff",
    "adjoe_ratio", "adjde_ratio", "barthag_ratio", "sos_ratio", "wab_ratio", "adjt_ratio",
]

FEATURE_LABELS = {
    "adjoe_diff":    "Adj. Off. Eff. Diff",
    "adjde_diff":    "Adj. Def. Eff. Diff",
    "barthag_diff":  "Barthag Diff",
    "sos_diff":      "Strength of Schedule Diff",
    "wab_diff":      "Wins Above Bubble Diff",
    "adjt_diff":     "Adj. Tempo Diff",
    "adjoe_ratio":   "Adj. Off. Eff. Ratio",
    "adjde_ratio":   "Adj. Def. Eff. Ratio",
    "barthag_ratio": "Barthag Ratio",
    "sos_ratio":     "SoS Ratio",
    "wab_ratio":     "WAB Ratio",
    "adjt_ratio":    "Tempo Ratio",
}

TARGET = "favorite_win_flag"


@st.cache_resource(show_spinner=False)
def _get_trained_xgb():
    """Train XGBoost on full training data, return (model, scaler_unused)."""
    try:
        from xgboost import XGBClassifier
        train = pd.read_csv(DATA_DIR / "ml_training_data.csv")
        train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
        train = train.dropna(subset=[TARGET] + FEATURES)
        X = train[FEATURES].values
        y = train[TARGET].values
        clf = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", verbosity=0, random_state=42
        )
        clf.fit(X, y)
        return clf
    except Exception as e:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def compute_shap_for_matchup(team_a: str, team_b: str) -> dict:
    """
    Compute SHAP values for the (team_a, team_b) matchup.

    Returns:
        expected_value    : float
        shap_values       : list[float]
        feature_names     : list[str]
        feature_values    : list[float]
        top5_positive     : list[(label, shap_val, feat_val)]
        top5_negative     : list[(label, shap_val, feat_val)]
        predicted_prob    : float
        waterfall_fig     : matplotlib Figure (or None)
        error             : str | None
    """
    try:
        import shap
    except ImportError:
        return {"error": "shap not installed — run: pip install shap"}

    clf = _get_trained_xgb()
    if clf is None:
        return {"error": "XGBoost model could not be trained (check data files)."}

    # Build matchup feature vector
    infer = pd.read_csv(DATA_DIR / "ml_inference_data_2026.csv")
    row = infer[
        ((infer["team_a"] == team_a) & (infer["team_b"] == team_b)) |
        ((infer["team_a"] == team_b) & (infer["team_b"] == team_a))
    ]
    if len(row) == 0:
        return {"error": f"Matchup {team_a} vs {team_b} not found in inference data."}

    row = row.iloc[0]
    flipped = row["team_a"] == team_b  # if stored as B vs A
    X_vec = row[FEATURES].values.astype(float).reshape(1, -1)
    if flipped:
        # Negate diffs, invert ratios so team_a is always the "perspective" team
        for i, feat in enumerate(FEATURES):
            if "_diff" in feat:
                X_vec[0, i] = -X_vec[0, i]
            elif "_ratio" in feat:
                X_vec[0, i] = 1.0 / (X_vec[0, i] + 1e-9)

    pred_prob = float(clf.predict_proba(X_vec)[0, 1])

    # SHAP explainer
    explainer  = shap.TreeExplainer(clf)
    shap_vals  = explainer.shap_values(X_vec)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # class=1 for binary
    sv = shap_vals[0]  # shape (n_features,)
    ev = float(explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__")
               else explainer.expected_value)
    fv = X_vec[0]

    labels = [FEATURE_LABELS.get(f, f) for f in FEATURES]

    sorted_idx = np.argsort(sv)[::-1]
    top5_pos = [(labels[i], float(sv[i]), float(fv[i]))
                for i in sorted_idx if sv[i] > 0][:5]
    top5_neg = [(labels[i], float(sv[i]), float(fv[i]))
                for i in sorted_idx[::-1] if sv[i] < 0][:5]

    # Waterfall plot
    fig = None
    try:
        matplotlib.use("Agg")
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#1e2a3a")
        ax.set_facecolor("#1e2a3a")

        # Sort by absolute SHAP value for waterfall
        order = np.argsort(np.abs(sv))[::-1][:8]
        y_pos = np.arange(len(order))
        colors = ["#43A047" if sv[i] > 0 else "#E53935" for i in order]
        bars = ax.barh(y_pos, sv[order], color=colors, alpha=0.85, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([labels[i] for i in order], color="#e8ecf0", fontsize=9)
        ax.set_xlabel("SHAP value (→ favours team A)", color="#9ab", fontsize=9)
        ax.set_title(f"SHAP — why model favours {team_a}", color="#e8ecf0", fontsize=11, pad=10)
        ax.axvline(0, color="#9ab", linewidth=0.8)
        ax.tick_params(colors="#9ab")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2e4060")
        # Annotate bars
        for bar, i in zip(bars, order):
            w = bar.get_width()
            ax.text(w + (0.002 if w >= 0 else -0.002), bar.get_y() + bar.get_height()/2,
                    f"{w:+.3f}", va="center",
                    ha="left" if w >= 0 else "right",
                    color="#e8ecf0", fontsize=7.5)
        plt.tight_layout()
    except Exception:
        fig = None

    return {
        "expected_value":  ev,
        "shap_values":     sv.tolist(),
        "feature_names":   FEATURES,
        "feature_labels":  labels,
        "feature_values":  fv.tolist(),
        "top5_positive":   top5_pos,
        "top5_negative":   top5_neg,
        "predicted_prob":  pred_prob,
        "waterfall_fig":   fig,
        "error":           None,
    }

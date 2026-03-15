"""
utils/style.py
Color palettes, conference colors, and CSS injection helpers.
"""
import streamlit as st

# ── Conference colour palette ─────────────────────────────────────────────────
CONF_COLORS = {
    "ACC":           "#003087",
    "Big Ten":       "#0088CE",
    "Big 12":        "#003366",
    "SEC":           "#F1C022",
    "Big East":      "#002F6C",
    "Pac-12":        "#003057",
    "WCC":           "#6D1945",
    "Mountain West": "#005CA9",
    "Atlantic 10":   "#6F2C3F",
    "American":      "#C8102E",
    "MVC":           "#6B2D90",
    "MAC":           "#003087",
    "Southland":     "#007A4D",
    "Ivy League":    "#8B0000",
    "SWAC":          "#006400",
    "MEAC":          "#800000",
}
DEFAULT_CONF_COLOR = "#5A5A5A"

# ── Round colours (blue → gold) ───────────────────────────────────────────────
ROUND_COLORS = {
    "Round of 64":  "#1565C0",
    "Round of 32":  "#1976D2",
    "Sweet 16":     "#388E3C",
    "Elite 8":      "#F57C00",
    "Final Four":   "#E64A19",
    "Champion":     "#FFD700",
}

# ── Probability gradient ──────────────────────────────────────────────────────
def prob_to_color(p: float) -> str:
    """Map probability 0-1 to a hex color (red → yellow → green)."""
    p = max(0.0, min(1.0, p))
    if p < 0.5:
        r, g, b = 220, int(220 * (p / 0.5)), 0
    else:
        r, g, b = int(220 * ((1 - p) / 0.5)), 180, 0
    return f"#{r:02x}{g:02x}{b:02x}"


def conf_color(conf: str) -> str:
    return CONF_COLORS.get(conf, DEFAULT_CONF_COLOR)


# ── CSS injection ─────────────────────────────────────────────────────────────
def inject_css() -> None:
    """Inject global custom CSS into the Streamlit app."""
    st.markdown("""
    <style>
    /* ── Global ─────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Sidebar ────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 100%);
        color: #e8ecf0;
    }
    [data-testid="stSidebar"] .stMarkdown p { color: #c9d4de; }
    [data-testid="stSidebar"] label { color: #e8ecf0 !important; }

    /* ── Metric cards ───────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, #1e2a3a, #243040);
        border: 1px solid #2e4060;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.35);
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #FFD700;
        line-height: 1.1;
    }
    .metric-card .label {
        font-size: 0.78rem;
        color: #9ab;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }

    /* ── Team name pill ─────────────────────────── */
    .seed-badge {
        display: inline-block;
        background: #1565C0;
        color: white;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 20px;
        margin-right: 6px;
    }
    .conf-badge {
        display: inline-block;
        background: #2e4060;
        color: #c9d4de;
        font-size: 0.72rem;
        padding: 2px 8px;
        border-radius: 20px;
    }

    /* ── Section headers ────────────────────────── */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #e8ecf0;
        border-bottom: 2px solid #1565C0;
        padding-bottom: 6px;
        margin-bottom: 14px;
    }

    /* ── Matchup bar ────────────────────────────── */
    .prob-bar-container {
        display: flex;
        align-items: center;
        border-radius: 8px;
        overflow: hidden;
        height: 36px;
        font-weight: 700;
        font-size: 1rem;
    }
    .prob-bar-a {
        background: linear-gradient(90deg, #1565C0, #1976D2);
        color: white;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 10px;
        transition: width 0.4s ease;
    }
    .prob-bar-b {
        background: linear-gradient(90deg, #C62828, #D32F2F);
        color: white;
        display: flex;
        align-items: center;
        padding-left: 10px;
        transition: width 0.4s ease;
    }

    /* ── Upset tag ──────────────────────────────── */
    .upset-badge {
        background: linear-gradient(135deg, #E64A19, #BF360C);
        color: white;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ── Edge advantage chip ───────────────────── */
    .edge-positive { color: #43A047; font-weight: 600; }
    .edge-negative { color: #E53935; font-weight: 600; }
    .edge-neutral  { color: #9ab; }

    /* ── Page title ─────────────────────────────── */
    .page-title {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1976D2, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .page-subtitle {
        color: #9ab;
        font-size: 0.9rem;
        margin-bottom: 24px;
    }

    /* ── Bracket slot ───────────────────────────── */
    .bracket-slot {
        background: #1e2a3a;
        border-left: 3px solid #1565C0;
        border-radius: 0 6px 6px 0;
        padding: 4px 8px;
        margin: 1px 0;
        font-size: 0.8rem;
        display: flex;
        justify-content: space-between;
    }

    /* ── Hide Streamlit default chrome ─────────── */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="page-title">🏀 {title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def metric_card(value: str, label: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="value">{value}</div>
        <div class="label">{label}</div>
    </div>"""


def seed_badge(seed: int) -> str:
    return f'<span class="seed-badge">#{seed}</span>'


def conf_badge(conf: str) -> str:
    color = conf_color(conf)
    return f'<span class="conf-badge" style="background:{color}20;border:1px solid {color}40;">{conf}</span>'

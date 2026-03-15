"""
utils/style.py
Color palettes, conference colors, and CSS injection helpers.
Aligned with assets/style.css design tokens — March Madness ML 2026.
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
DEFAULT_CONF_COLOR = "#4a6070"

# ── Round colours (blue → gold) ───────────────────────────────────────────────
ROUND_COLORS = {
    "Round of 64":  "#1260c4",
    "Round of 32":  "#1e8aff",
    "Sweet 16":     "#2da85a",
    "Elite 8":      "#e67500",
    "Final Four":   "#e64a19",
    "Champion":     "#f5c400",
}

# ── Probability gradient ──────────────────────────────────────────────────────
def prob_to_color(p: float) -> str:
    """Map probability 0–1 to a hex color (red → yellow → green)."""
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
    """Inject global custom CSS into the Streamlit app.

    Loads Barlow Condensed (display) + DM Sans (body) from Google Fonts and
    applies the full March Madness 2026 design system inline.  Heavy lifting
    (card hover states, animations, responsive rules) lives in assets/style.css;
    this block covers Streamlit-specific chrome and core component styles.
    """
    st.markdown("""
    <style>
    /* ── Google Fonts ───────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800;900&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&display=swap');

    /* ── Design tokens ──────────────────────────── */
    :root {
        --bg-primary:    #07111c;
        --bg-secondary:  #0d1b2a;
        --bg-card:       #111f2e;
        --bg-card-hover: #172537;
        --border-subtle: rgba(46, 82, 108, 0.55);
        --border-focus:  #1e8aff;
        --text-primary:  #dce9f5;
        --text-secondary:#6e8fa8;
        --blue-bright:   #1e8aff;
        --blue-dark:     #1260c4;
        --gold:          #f5c400;
        --orange:        #ff5722;
        --green:         #3fba6a;
        --red:           #f03a3a;
        --radius-sm:     8px;
        --radius-md:     14px;
        --shadow-card:   0 4px 24px rgba(0,0,0,0.45), 0 1px 0 rgba(255,255,255,0.035) inset;
        --glow-blue:     0 0 18px rgba(30,138,255,0.28), 0 0 48px rgba(30,138,255,0.10);
        --glow-gold:     0 0 18px rgba(245,196,0,0.30),  0 0 48px rgba(245,196,0,0.10);
        --transition:    0.22s cubic-bezier(0.4, 0, 0.2, 1);
        --font-display:  'Barlow Condensed', system-ui, sans-serif;
        --font-body:     'DM Sans', system-ui, sans-serif;
    }

    /* ── Global ─────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: var(--font-body);
        color: var(--text-primary);
        -webkit-font-smoothing: antialiased;
    }

    /* ── Sidebar ────────────────────────────────── */
    [data-testid="stSidebar"] {
        background:
            linear-gradient(160deg, rgba(18,96,196,0.06) 0%, transparent 50%),
            linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        border-right: 1px solid var(--border-subtle);
        box-shadow: 4px 0 32px rgba(0,0,0,0.35);
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: var(--text-secondary);
        font-size: 0.875rem;
        line-height: 1.6;
    }
    [data-testid="stSidebar"] label {
        color: var(--text-primary) !important;
        font-weight: 500;
        letter-spacing: 0.01em;
    }

    /* ── Metric cards ───────────────────────────── */
    .metric-card {
        background:
            linear-gradient(145deg, rgba(255,255,255,0.04) 0%, transparent 60%),
            var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 20px 24px;
        text-align: center;
        box-shadow: var(--shadow-card);
        position: relative;
        overflow: hidden;
        transition: border-color var(--transition), box-shadow var(--transition),
                    transform var(--transition);
    }
    .metric-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 50%;
        transform: translateX(-50%);
        width: 40%; height: 2px;
        background: linear-gradient(90deg, transparent, var(--gold), transparent);
        opacity: 0.7;
    }
    .metric-card:hover {
        border-color: rgba(245,196,0,0.35);
        box-shadow: var(--shadow-card), var(--glow-gold);
        transform: translateY(-2px);
    }
    .metric-card .value {
        font-family: var(--font-display);
        font-size: 2.6rem;
        font-weight: 900;
        color: var(--gold);
        line-height: 1.0;
        letter-spacing: -0.01em;
        text-shadow: var(--glow-gold);
    }
    .metric-card .label {
        font-size: 0.72rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-top: 6px;
        font-weight: 600;
    }

    /* ── Badges ─────────────────────────────────── */
    .seed-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, var(--blue-dark), var(--blue-bright));
        color: white;
        font-family: var(--font-display);
        font-weight: 800;
        font-size: 0.70rem;
        padding: 2px 9px;
        border-radius: 20px;
        margin-right: 7px;
        letter-spacing: 0.04em;
        box-shadow: 0 1px 6px rgba(30,138,255,0.30);
    }
    .conf-badge {
        display: inline-flex;
        align-items: center;
        font-size: 0.70rem;
        padding: 2px 9px;
        border-radius: 20px;
        letter-spacing: 0.02em;
    }

    /* ── Section headers ────────────────────────── */
    .section-header {
        font-family: var(--font-display);
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--text-primary);
        border-bottom: 2px solid transparent;
        border-image: linear-gradient(90deg, var(--blue-bright), transparent) 1;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* ── Probability bar ─────────────────────────── */
    .prob-bar-container {
        display: flex;
        align-items: stretch;
        border-radius: var(--radius-sm);
        overflow: hidden;
        height: 38px;
        font-family: var(--font-display);
        font-weight: 800;
        font-size: 1.05rem;
        letter-spacing: 0.02em;
        box-shadow: 0 2px 12px rgba(0,0,0,0.40);
    }
    .prob-bar-a {
        background: linear-gradient(90deg, var(--blue-dark), var(--blue-bright));
        color: white;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 12px;
        transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        text-shadow: 0 1px 4px rgba(0,0,0,0.30);
    }
    .prob-bar-b {
        background: linear-gradient(90deg, #b71c1c, #e53935);
        color: white;
        display: flex;
        align-items: center;
        padding-left: 12px;
        transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        text-shadow: 0 1px 4px rgba(0,0,0,0.30);
    }

    /* ── Upset badge ─────────────────────────────── */
    .upset-badge {
        background: linear-gradient(135deg, #c84115, var(--orange));
        color: white;
        font-family: var(--font-display);
        font-size: 0.70rem;
        font-weight: 800;
        padding: 3px 12px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        box-shadow: 0 2px 10px rgba(255,87,34,0.35);
    }

    /* ── Edge indicators ────────────────────────── */
    .edge-positive {
        color: var(--green);
        font-weight: 600;
        text-shadow: 0 0 8px rgba(63,186,106,0.35);
    }
    .edge-negative {
        color: var(--red);
        font-weight: 600;
        text-shadow: 0 0 8px rgba(240,58,58,0.35);
    }
    .edge-neutral { color: var(--text-secondary); }

    /* ── Page title ─────────────────────────────── */
    .page-title {
        font-family: var(--font-display);
        font-size: 2.4rem;
        font-weight: 900;
        letter-spacing: -0.01em;
        line-height: 1.1;
        background: linear-gradient(100deg,
            var(--blue-bright) 0%,
            var(--gold) 55%,
            var(--orange) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 4px;
    }
    .page-subtitle {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-bottom: 28px;
        font-weight: 400;
        letter-spacing: 0.01em;
    }

    /* ── Bracket slot ───────────────────────────── */
    .bracket-slot {
        background: var(--bg-card);
        border-left: 3px solid var(--blue-dark);
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        padding: 5px 12px;
        margin: 2px 0;
        font-size: 0.80rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: border-color var(--transition), background var(--transition),
                    transform var(--transition);
    }
    .bracket-slot:hover {
        border-left-color: var(--gold);
        background: var(--bg-card-hover);
        transform: translateX(2px);
    }
    .bracket-slot.winner {
        border-left-color: var(--green);
        font-weight: 600;
        background: linear-gradient(90deg, rgba(63,186,106,0.08), transparent);
    }

    /* ── Hide Streamlit chrome ──────────────────── */
    #MainMenu, footer, header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ── HTML component helpers ────────────────────────────────────────────────────

def page_header(title: str, subtitle: str = "") -> None:
    """Render a gradient page title with an optional subtitle."""
    st.markdown(f'<div class="page-title">🏀 {title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def metric_card(value: str, label: str) -> str:
    """Return an HTML metric card string for use with st.markdown."""
    return f"""
    <div class="metric-card">
        <div class="value">{value}</div>
        <div class="label">{label}</div>
    </div>"""


def seed_badge(seed: int) -> str:
    """Return an HTML seed badge string (e.g. '#1')."""
    return f'<span class="seed-badge">#{seed}</span>'


def conf_badge(conf: str) -> str:
    """Return an HTML conference badge with the conference's brand color."""
    color = conf_color(conf)
    return (
        f'<span class="conf-badge" '
        f'style="background:{color}22; border:1px solid {color}44; color:#dce9f5;">'
        f'{conf}</span>'
    )


def round_badge(round_name: str) -> str:
    """Return an HTML round badge using the ROUND_COLORS palette."""
    color = ROUND_COLORS.get(round_name, DEFAULT_CONF_COLOR)
    is_champ = round_name == "Champion"
    text_color = "#07111c" if is_champ else "white"
    glow = "box-shadow:0 0 14px rgba(245,196,0,0.40);" if is_champ else ""
    return (
        f'<span style="display:inline-flex;align-items:center;'
        f'font-family:\'Barlow Condensed\',sans-serif;font-weight:800;'
        f'font-size:0.70rem;letter-spacing:0.06em;text-transform:uppercase;'
        f'padding:2px 11px;border-radius:20px;color:{text_color};'
        f'background:{color};{glow}">'
        f'{round_name}</span>'
    )


def prob_bar(prob_a: float, name_a: str, name_b: str) -> str:
    """Return an HTML win-probability split bar."""
    pct_a = round(prob_a * 100)
    pct_b = 100 - pct_a
    return f"""
    <div class="prob-bar-container">
        <div class="prob-bar-a" style="width:{pct_a}%">{name_a} {pct_a}%</div>
        <div class="prob-bar-b" style="width:{pct_b}%">{pct_b}% {name_b}</div>
    </div>"""


def champ_pill(text: str) -> str:
    """Return a gold champion-probability pill."""
    return (
        f'<span style="display:inline-flex;align-items:center;gap:5px;'
        f'background:linear-gradient(90deg,#6a5100,#a87d00,#f5c400);'
        f'color:#07111c;font-family:\'Barlow Condensed\',sans-serif;'
        f'font-weight:900;font-size:0.82rem;padding:4px 14px;'
        f'border-radius:20px;letter-spacing:0.06em;text-transform:uppercase;'
        f'box-shadow:0 0 18px rgba(245,196,0,0.30);">'
        f'🏆 {text}</span>'
    )
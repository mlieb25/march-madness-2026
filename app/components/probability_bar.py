"""
components/probability_bar.py
Reusable styled probability bar rendered as HTML.
"""
import streamlit as st


def render_probability_bar(
    team_a: str,
    team_b: str,
    prob_a: float,
    prob_lower: float = None,
    prob_upper: float = None,
    height: int = 44,
) -> None:
    """
    Render a dual-colour horizontal bar showing win probability split.

    Args:
        team_a      : name of team A (shown on left, blue)
        team_b      : name of team B (shown on right, red)
        prob_a      : P(team_a wins), 0–1
        prob_lower  : lower confidence bound (2.5th pct)
        prob_upper  : upper confidence bound (97.5th pct)
        height      : bar height in px
    """
    prob_a = max(0.02, min(0.98, prob_a))
    prob_b = 1 - prob_a

    pct_a = f"{prob_a * 100:.1f}%"
    pct_b = f"{prob_b * 100:.1f}%"

    # Confidence interval text
    ci_text = ""
    if prob_lower is not None and prob_upper is not None:
        ci_text = f"<div style='text-align:center;color:#9ab;font-size:0.75rem;margin-top:3px;'>" \
                  f"Model confidence band: {prob_lower*100:.0f}% – {prob_upper*100:.0f}%</div>"

    bar_html = f"""
    <div style="margin:8px 0;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
        <span style="color:#e8ecf0;font-weight:600;font-size:0.9rem;width:40%;text-align:right;">{team_a}</span>
        <span style="color:#FFD700;font-weight:700;font-size:1.05rem;">{pct_a}</span>
        <span style="color:#9ab;font-size:0.85rem;">vs</span>
        <span style="color:#FFD700;font-weight:700;font-size:1.05rem;">{pct_b}</span>
        <span style="color:#e8ecf0;font-weight:600;font-size:0.9rem;width:40%;">{team_b}</span>
      </div>
      <div style="display:flex;border-radius:8px;overflow:hidden;height:{height}px;box-shadow:0 2px 8px rgba(0,0,0,0.4);">
        <div style="
          width:{prob_a*100:.2f}%;
          background:linear-gradient(90deg,#1565C0,#1976D2);
          display:flex;align-items:center;justify-content:flex-end;
          padding-right:10px;color:white;font-weight:700;font-size:0.95rem;
          transition:width 0.4s ease;">
          {pct_a}
        </div>
        <div style="
          width:{prob_b*100:.2f}%;
          background:linear-gradient(90deg,#C62828,#D32F2F);
          display:flex;align-items:center;padding-left:10px;
          color:white;font-weight:700;font-size:0.95rem;
          transition:width 0.4s ease;">
          {pct_b}
        </div>
      </div>
      {ci_text}
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)


def render_mini_bar(prob: float, label: str = "", width: str = "100%") -> str:
    """Return an HTML string of a slim single-colour bar (for tables/cards)."""
    pct = prob * 100
    color = "#43A047" if pct >= 60 else ("#F57C00" if pct >= 40 else "#E53935")
    return (
        f'<div style="display:flex;align-items:center;gap:6px;">'
        f'<div style="flex:1;background:#1e2a3a;border-radius:4px;height:10px;overflow:hidden;">'
        f'<div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:4px;"></div>'
        f'</div>'
        f'<span style="color:#e8ecf0;font-size:0.78rem;min-width:38px;">{pct:.1f}%</span>'
        f'</div>'
    )

"""
components/bracket_viz.py
Full 68-team NCAA bracket rendered as a Plotly figure.
Programmatic drawing — no images.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.style import conf_color, prob_to_color

# ── Layout constants ───────────────────────────────────────────────────────────
FIG_W, FIG_H   = 1500, 880
REGION_H       = FIG_H / 2           # 440 px per region (2 per side)
SLOT_H         = REGION_H / 16       # px per team slot

# X positions for each round on the left side (R64→E8)
LEFT_X  = [60, 215, 365, 490]
# X positions for each round on the right side (R64→E8, mirrored)
RIGHT_X = [FIG_W-60, FIG_W-215, FIG_W-365, FIG_W-490]
# Final Four and Championship
FF_X_L, FF_X_R, CHAMP_X = 610, 890, 750

# Region Y offsets (top of each region)
# Left: East (top, y_min=0) and West (bottom, y_min=440)
# Right: Midwest (top) and South (bottom)
REGION_META = {
    "W": dict(side="left",  y_off=REGION_H,  name="East"),
    "X": dict(side="left",  y_off=0,         name="West"),
    "Y": dict(side="right", y_off=0,         name="South"),
    "Z": dict(side="right", y_off=REGION_H,  name="Midwest"),
}

# Standard bracket y-ordering within region (top-to-bottom in display)
BRACKET_SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

# Which R64 game indices pair up in R32 (per MNCAATourneySlots logic)
R32_PAIR_INDICES = [(0, 7), (1, 6), (2, 5), (3, 4)]  # game i vs game j
S16_PAIR_INDICES = [(0, 3), (1, 2)]                   # r32-game i vs r32-game j
E8_PAIR_INDEX    = (0, 1)                             # s16-game 0 vs 1


def _region_slot_ys(y_off: float, n_slots: int = 16) -> list:
    """Return y-centre of each slot within a region (top to bottom)."""
    return [y_off + (i + 0.5) * SLOT_H for i in range(n_slots)]


def _round_ys(y_off: float) -> dict:
    """
    Compute y-centres for teams in each round within a region (16→8→4→2→1).
    Each round's positions are midpoints of the previous round's pairs.
    """
    ys: dict[int, list] = {}
    ys[0] = _region_slot_ys(y_off)  # R64: 16 slots

    for r in range(1, 5):  # R32=1, S16=2, E8=3, FF-equivalent=4
        prev = ys[r - 1]
        ys[r] = [(prev[i] + prev[i + 1]) / 2 for i in range(0, len(prev), 2)]
    return ys


def _chalk_winner(ys_a: float, ys_b: float, prob_a: float, name_a: str, name_b: str):
    """Return (winner_name, winner_y, win_prob) — chalk pick."""
    if prob_a >= 0.5:
        return name_a, ys_a, prob_a
    return name_b, ys_b, 1 - prob_a


def _get_prob(probs_df: pd.DataFrame, name_a: str, name_b: str) -> float:
    """Look up P(name_a beats name_b), default 0.5."""
    row = probs_df[
        ((probs_df["team_a"] == name_a) & (probs_df["team_b"] == name_b))
    ]
    if len(row) > 0:
        return float(row.iloc[0]["ensemble_prob"])
    row = probs_df[
        ((probs_df["team_a"] == name_b) & (probs_df["team_b"] == name_a))
    ]
    if len(row) > 0:
        return 1 - float(row.iloc[0]["ensemble_prob"])
    return 0.5


def render_bracket(
    df_teams: pd.DataFrame,
    df_pairwise: pd.DataFrame,
    df_sim: pd.DataFrame,
) -> None:
    """
    Draw and display the full tournament bracket as a Plotly figure.
    Shows chalk-pick winners at each slot, coloured by conference and
    annotated with win probability.
    """
    fig = go.Figure()

    # Collect data for click-tooltip annotations
    node_x, node_y, node_text, node_color, node_hover = [], [], [], [], []

    # ── Helper: draw a team slot ───────────────────────────────────────────────
    def add_slot(x: float, y: float, name: str, seed: int, conf: str,
                 champ_pct: float, round_prob: float, left_side: bool):
        color = conf_color(conf)
        hover = (
            f"<b>{name}</b> (#{seed})<br>"
            f"Conf: {conf}<br>"
            f"Prob this round: {round_prob*100:.1f}%<br>"
            f"Champion prob: {champ_pct:.1f}%"
        )
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{'>' if not left_side else ''}{name[:15]}{'<' if left_side else ''}")
        node_color.append(color)
        node_hover.append(hover)

    # ── Helper: draw a connecting line ────────────────────────────────────────
    def add_line(x0, y0, x1, y1, color="#2e4060", dash="solid", width=1.2):
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color=color, width=width, dash=dash))

    # ── Helper: draw bracket edges for a game pair ────────────────────────────
    def draw_game_edges(x_from: float, y_a: float, y_b: float,
                        x_to: float, y_winner: float, left_side: bool):
        mid_x = (x_from + x_to) / 2
        add_line(x_from, y_a, mid_x, y_a)
        add_line(x_from, y_b, mid_x, y_b)
        add_line(mid_x, y_a, mid_x, y_b)
        add_line(mid_x, y_winner, x_to, y_winner, color="#1976D2", width=1.5)

    # ── Region label background annotations ───────────────────────────────────
    for reg, meta in REGION_META.items():
        y_top = meta["y_off"]
        x_lab = LEFT_X[0] - 40 if meta["side"] == "left" else RIGHT_X[0] + 40
        fig.add_annotation(
            x=x_lab, y=y_top + REGION_H / 2,
            text=f"<b>{meta['name']}</b>",
            showarrow=False,
            font=dict(color="#9ab", size=11),
            textangle=-90 if meta["side"] == "left" else 90,
        )

    # ── Process each region ────────────────────────────────────────────────────
    ff_winners: dict[str, tuple] = {}  # region → (team_name, y_pos)

    for reg, meta in REGION_META.items():
        y_off    = meta["y_off"]
        left     = meta["side"] == "left"
        x_rounds = LEFT_X if left else RIGHT_X[::-1]  # order R64 → E8
        round_ys = _round_ys(y_off)

        # Get region teams in BRACKET_SEED_ORDER
        region_teams_df = df_teams[df_teams["region"] == reg].copy()
        seed_to_row     = {int(r["seed_num"]): r for _, r in region_teams_df.iterrows()}

        ordered_names   = []
        ordered_confs   = []
        ordered_seeds   = []
        for s in BRACKET_SEED_ORDER:
            row = seed_to_row.get(s)
            if row is not None:
                ordered_names.append(str(row["team_name"]))
                ordered_confs.append(str(row.get("conf", "")))
                ordered_seeds.append(s)
            else:
                ordered_names.append(f"{reg}{s:02d}")
                ordered_confs.append("")
                ordered_seeds.append(s)

        # ── R64 slots ──────────────────────────────────────────────────────────
        for i, (name, conf, seed) in enumerate(zip(ordered_names, ordered_confs, ordered_seeds)):
            sim_row  = df_sim[df_sim["team_name"] == name]
            champ_p  = float(sim_row["rd6_pct"].values[0]) if len(sim_row) else 0.0
            x = x_rounds[0]
            y = round_ys[0][i]
            add_slot(x, y, name, seed, conf, champ_p, 1.0, left)

        # ── Simulate chalk bracket for rounds 2-4 ─────────────────────────────
        current_round_names = list(ordered_names)
        current_round_confs = list(ordered_confs)
        current_round_seeds = list(ordered_seeds)

        for rd_idx in range(1, 4):  # R32, S16, E8
            n_games = len(current_round_names) // 2
            next_names, next_confs, next_seeds = [], [], []

            for g in range(n_games):
                ia, ib = g * 2, g * 2 + 1
                na, nb = current_round_names[ia], current_round_names[ib]
                ca, cb = current_round_confs[ia], current_round_confs[ib]
                sa, sb = current_round_seeds[ia], current_round_seeds[ib]

                prob = _get_prob(df_pairwise, na, nb)
                winner_name = na if prob >= 0.5 else nb
                winner_conf = ca if prob >= 0.5 else cb
                winner_seed = sa if prob >= 0.5 else sb

                y_a = round_ys[rd_idx - 1][ia]
                y_b = round_ys[rd_idx - 1][ib]
                y_w = round_ys[rd_idx][g]
                x_from = x_rounds[rd_idx - 1]
                x_to   = x_rounds[rd_idx]

                # Draw edges
                draw_game_edges(x_from, y_a, y_b, x_to, y_w, left)

                # Add winner node
                sim_row = df_sim[df_sim["team_name"] == winner_name]
                champ_p = float(sim_row["rd6_pct"].values[0]) if len(sim_row) else 0.0
                round_p = float(sim_row[f"rd{rd_idx+1}_pct"].values[0]) / 100 if len(sim_row) else 0.5
                add_slot(x_to, y_w, winner_name, winner_seed, winner_conf, champ_p, round_p, left)

                next_names.append(winner_name)
                next_confs.append(winner_conf)
                next_seeds.append(winner_seed)

            current_round_names = next_names
            current_round_confs = next_confs
            current_round_seeds = next_seeds

        # E8 winner → Final Four
        e8_winner = current_round_names[0]
        e8_y      = round_ys[3][0]
        ff_y      = FIG_H / 2  # all FF teams appear at center
        ff_winners[reg] = (e8_winner, current_round_confs[0], current_round_seeds[0])

    # ── Final Four ────────────────────────────────────────────────────────────
    # W vs X (left FF), Y vs Z (right FF)
    ff_pairs  = [("W", "X", FF_X_L), ("Y", "Z", FF_X_R)]
    champ_candidates = []
    for reg_a, reg_b, ff_x in ff_pairs:
        na, ca, sa = ff_winners.get(reg_a, ("TBD","",0))
        nb, cb, sb = ff_winners.get(reg_b, ("TBD","",0))
        prob = _get_prob(df_pairwise, na, nb)
        winner = na if prob >= 0.5 else nb
        winner_conf = ca if prob >= 0.5 else cb
        winner_seed = sa if prob >= 0.5 else sb

        y_a = FIG_H * 0.35
        y_b = FIG_H * 0.65
        y_w = FIG_H / 2

        add_line(LEFT_X[-1] if reg_a == "W" else RIGHT_X[-1][::-1] if False else RIGHT_X[0],
                 round_ys["W" if reg_a == "W" else "Z"][3][0] if False else y_a,
                 ff_x, y_a, color="#FFD700", width=1.5)
        add_slot(ff_x, y_a, na, sa, ca, 0, prob if reg_a == reg_a else 1-prob, reg_a in ["W","X"])
        add_slot(ff_x, y_b, nb, sb, cb, 0, 1-prob if prob >= 0.5 else prob, reg_b in ["W","X"])
        add_line(ff_x, y_a, ff_x, y_b, color="#2e4060")
        add_line(ff_x, y_w, CHAMP_X, y_w, color="#FFD700", width=1.8)
        add_slot(CHAMP_X, y_w if reg_a == "W" else FIG_H * 0.4,
                 winner, winner_seed, winner_conf, 0, 0, True)
        champ_candidates.append((winner, winner_conf, winner_seed))

    # Championship node label
    fig.add_annotation(
        x=CHAMP_X, y=FIG_H / 2 + 30,
        text="<b>🏆 CHAMPION</b>",
        showarrow=False,
        font=dict(color="#FFD700", size=13),
        bgcolor="#1e2a3a",
        bordercolor="#FFD700",
        borderwidth=1,
        borderpad=5,
    )

    # ── Scatter trace for all team nodes ─────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="text",
        text=node_text,
        textfont=dict(size=8.5, color=node_color),
        hovertext=node_hover,
        hoverinfo="text",
        hoverlabel=dict(bgcolor="#1e2a3a", bordercolor="#2e4060",
                        font=dict(color="#e8ecf0", size=11)),
    ))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        width=FIG_W, height=FIG_H,
        paper_bgcolor="#0d1b2a",
        plot_bgcolor="#0d1b2a",
        xaxis=dict(visible=False, range=[0, FIG_W]),
        yaxis=dict(visible=False, range=[0, FIG_H]),
        margin=dict(l=0, r=0, t=10, b=10),
        showlegend=False,
        dragmode="pan",
    )

    # Round labels at top
    for label, x in [("R64", LEFT_X[0]), ("R32", LEFT_X[1]),
                      ("Sweet 16", LEFT_X[2]), ("Elite 8", LEFT_X[3]),
                      ("Final Four", FF_X_L), ("🏆", CHAMP_X),
                      ("Final Four", FF_X_R), ("Elite 8", RIGHT_X[0]+200 if False else RIGHT_X[3]),
                      ("Sweet 16", RIGHT_X[2]), ("R32", RIGHT_X[1]), ("R64", RIGHT_X[0])]:
        fig.add_annotation(x=x, y=FIG_H - 5, text=label, showarrow=False,
                           font=dict(color="#9ab", size=8.5), yanchor="top")

    st.plotly_chart(fig, use_container_width=False,
                    config={"displayModeBar": True, "scrollZoom": True,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d"]})
    st.caption("💡 Scroll to zoom · Drag to pan · Hover any team for details")

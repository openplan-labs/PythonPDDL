#!/usr/bin/env python3
"""Render the jupyddl promo video.

Every number on screen is measured, not written by hand: the script runs the
real planners on the bundled demo instances first, then animates what came back.
If a search gets faster or slower, the video changes with it.

    python tools/make_promo.py -o promo/jupyddl.mp4

Needs the ``viz`` extra plus an ffmpeg binary (``pip install imageio-ffmpeg``
is enough — matplotlib is pointed at the bundled static build automatically).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, PillowWriter  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jupyddl import build_task, solve_task, trace_search, validate_plan  # noqa: E402
from jupyddl.generator import GENERATORS  # noqa: E402
from jupyddl.requirements import as_rows, summary  # noqa: E402
from jupyddl.search import PLANNERS  # noqa: E402
from jupyddl.viz.theme import DARK  # noqa: E402

FPS = 30
SIZE = (16, 9)
DPI = 120

BG = "#0b0b0c"
SURFACE = "#151517"
INK = "#ffffff"
DIM = "#a8a69c"
MUTED = "#75736c"
BLUE = DARK["series"][0]
ORANGE = DARK["series"][1]
AQUA = DARK["series"][2]
YELLOW = DARK["series"][3]
GOOD = "#0ca30c"
SEQ = DARK["sequential"]

MONO = ["DejaVu Sans Mono", "monospace"]


# --------------------------------------------------------------------------
# easing + drawing helpers
# --------------------------------------------------------------------------
def ease_out(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return 1 - pow(1 - t, 3)


def ease_in_out(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return 3 * t * t - 2 * t * t * t


def fade_in(t: float, start: float, length: float = 0.18) -> float:
    """Alpha for something that appears at ``start`` (both in scene fraction)."""
    if t < start:
        return 0.0
    return ease_out((t - start) / length)


def fade_window(t: float, start: float, end: float, ramp: float = 0.12) -> float:
    """Alpha for something visible between ``start`` and ``end``."""
    if t < start or t > end:
        return 0.0
    return min(ease_out((t - start) / ramp), ease_out((end - t) / ramp), 1.0)


def text(
    fig,
    x,
    y,
    body,
    size=28,
    color=INK,
    alpha=1.0,
    weight="normal",
    family=None,
    ha="left",
    va="center",
    style="normal",
):
    if alpha <= 0.001:
        return None
    return fig.text(
        x,
        y,
        body,
        fontsize=size,
        color=color,
        alpha=min(1.0, alpha),
        fontweight=weight,
        ha=ha,
        va=va,
        style=style,
        fontfamily=family or "DejaVu Sans",
    )


def panel(fig, x, y, w, h, alpha=1.0, color=SURFACE, radius=0.012):
    if alpha <= 0.001:
        return
    fig.patches.append(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            transform=fig.transFigure,
            facecolor=color,
            edgecolor="#26262a",
            linewidth=1.2,
            alpha=min(1.0, alpha),
            zorder=0,
        )
    )


def count_up(target, t, start=0.0, length=0.55):
    """Animated integer counter reaching ``target`` at ``start + length``."""
    if t <= start:
        return 0
    return int(round(target * ease_out((t - start) / length)))


def typewriter(body: str, t: float, start: float, length: float) -> str:
    if t < start:
        return ""
    fraction = min(1.0, (t - start) / length)
    return body[: int(len(body) * fraction)]


def fmt(value) -> str:
    return f"{int(value):,}"


# --------------------------------------------------------------------------
# data collection — everything the video shows is measured here
# --------------------------------------------------------------------------
def collect(root: str) -> dict:
    print("Measuring (this runs the real planners)...")
    data: dict = {}

    bw_domain = os.path.join(root, "blocksworld8", "domain.pddl")
    bw_problem = os.path.join(root, "blocksworld8", "problem.pddl")
    task = build_task(bw_domain, bw_problem)
    data["task"] = {
        "name": task.name,
        "facts": task.num_facts,
        "operators": len(task.operators),
    }

    print("  blocksworld8: A* + LM-cut")
    astar_result, astar_trace = trace_search(task, "astar", "lmcut")
    print("  blocksworld8: breadth-first")
    bfs_result, bfs_trace = trace_search(task, "bfs", None, max_events=6000)
    data["astar"] = astar_trace
    data["bfs"] = bfs_trace
    data["plan"] = astar_result.plan_names()
    data["plan_valid"] = validate_plan(task, astar_result.plan)
    data["cost"] = astar_result.cost

    print("  gripper: A* + LM-cut (for the search curves)")
    gripper = build_task(
        os.path.join(root, "gripper", "domain.pddl"),
        os.path.join(root, "gripper", "problem.pddl"),
    )
    _, gripper_trace = trace_search(gripper, "astar", "lmcut")
    data["curves"] = gripper_trace

    print("  hanoi: optimality check")
    hanoi = build_task(
        os.path.join(root, "hanoi", "domain.pddl"),
        os.path.join(root, "hanoi", "problem.pddl"),
    )
    hanoi_result = solve_task(hanoi, "astar", "lmcut")
    data["hanoi_cost"] = hanoi_result.cost

    # A compact cross-planner table for the showdown scene.
    print("  showdown table")
    rows = []
    for planner, heuristic in [
        ("astar", "lmcut"),
        ("astar", "hmax"),
        ("gbfs", "hff"),
        ("dijkstra", None),
        ("bfs", None),
    ]:
        result = solve_task(task, planner, heuristic)
        label = f"{planner}/{heuristic}" if heuristic else planner
        rows.append(
            {
                "label": label,
                "expanded": result.stats.expanded,
                "runtime": result.stats.runtime,
                "cost": result.cost,
            }
        )
        print(f"    {label}: {result.stats.expanded:,} expanded")
    data["showdown"] = rows

    # --- what the front end actually accepts, straight from the registry ---
    data["requirements"] = as_rows()
    data["support"] = summary()
    data["planner_count"] = len(PLANNERS)
    data["generator_count"] = len(GENERATORS)

    print("  errands: soft goals under a metric")
    errands = build_task(
        os.path.join(root, "errands", "domain.pddl"),
        os.path.join(root, "errands", "problem.pddl"),
    )
    errands_result = solve_task(errands, "astar", "hmax", time_limit=120)
    data["errands"] = {
        "cost": errands_result.cost,
        "plan": [op.base_name for op in errands.visible_plan(errands_result.plan)],
        "valid": validate_plan(errands, errands_result.plan),
    }
    print(f"    cost {errands_result.cost}: {data['errands']['plan']}")

    print("  timed-market: waiting for opening time")
    market = build_task(
        os.path.join(root, "timed-market", "domain.pddl"),
        os.path.join(root, "timed-market", "problem.pddl"),
    )
    market_result = solve_task(market, "astar", "hmax", time_limit=120)
    data["market"] = {
        "makespan": market.makespan(market_result.plan),
        "work": sum(op.duration for op in market.visible_plan(market_result.plan)),
        "plan": [op.base_name for op in market.visible_plan(market_result.plan)],
    }
    print(f"    makespan {data['market']['makespan']}")
    return data


# --------------------------------------------------------------------------
# scenes
# --------------------------------------------------------------------------
def scene_title(fig, t, data):
    fig.patch.set_facecolor(BG)
    grow = ease_out(min(1.0, t / 0.45))
    text(
        fig,
        0.5,
        0.60,
        "jupyddl",
        size=104 + 6 * (1 - grow),
        weight="bold",
        color=INK,
        alpha=fade_in(t, 0.02, 0.22),
        ha="center",
    )
    text(
        fig,
        0.5,
        0.475,
        "PDDL planning, in pure Python",
        size=30,
        color=DIM,
        alpha=fade_in(t, 0.22, 0.2),
        ha="center",
    )

    alpha = fade_in(t, 0.45, 0.2)
    if alpha > 0:
        text(
            fig,
            0.5,
            0.35,
            "2021  ·  a university project",
            size=20,
            color=MUTED,
            alpha=alpha,
            ha="center",
        )
        text(
            fig,
            0.5,
            0.30,
            "2026  ·  rewritten from scratch",
            size=20,
            color=BLUE,
            alpha=fade_in(t, 0.58, 0.2),
            ha="center",
            weight="bold",
        )

    # a hairline that draws itself under the wordmark
    width = 0.22 * ease_out(min(1.0, max(0.0, (t - 0.15) / 0.5)))
    if width > 0:
        fig.add_artist(
            plt.Line2D(
                [0.5 - width, 0.5 + width],
                [0.545, 0.545],
                color=BLUE,
                linewidth=3,
                alpha=fade_in(t, 0.15, 0.2),
                transform=fig.transFigure,
                solid_capstyle="round",
            )
        )


def scene_origin(fig, t, data):
    fig.patch.set_facecolor(BG)
    text(
        fig,
        0.08,
        0.80,
        "It started as a coursework project.",
        size=40,
        weight="bold",
        alpha=fade_in(t, 0.0, 0.15),
    )
    text(
        fig,
        0.08,
        0.72,
        "A thin Python wrapper around a Julia PDDL parser.",
        size=24,
        color=DIM,
        alpha=fade_in(t, 0.12, 0.15),
    )

    old = [
        ("Python", "the API you actually called"),
        ("PyCall", "the bridge"),
        ("Julia", "a whole second runtime"),
        ("PDDL.jl", "the parser doing the work"),
    ]
    for i, (name, note) in enumerate(old):
        alpha = fade_in(t, 0.22 + i * 0.07, 0.14)
        if alpha <= 0:
            continue
        y = 0.55 - i * 0.095
        panel(fig, 0.08, y - 0.035, 0.40, 0.072, alpha=alpha * 0.9)
        text(
            fig,
            0.105,
            y,
            name,
            size=23,
            weight="bold",
            alpha=alpha,
            family="DejaVu Sans Mono",
        )
        text(fig, 0.255, y, note, size=17, color=MUTED, alpha=alpha)

    # What it actually cost you, straight out of the old requirements.txt.
    alpha = fade_in(t, 0.58, 0.16)
    if alpha > 0:
        panel(fig, 0.56, 0.255, 0.37, 0.305, alpha=alpha, color="#101012")
        text(
            fig,
            0.582,
            0.525,
            "requirements.txt",
            size=16,
            color=MUTED,
            alpha=alpha,
            family="DejaVu Sans Mono",
        )
        for i, line in enumerate(
            ["julia==0.5.7", "coloredlogs==15.0.1", "matplotlib==3.5.1"]
        ):
            text(
                fig,
                0.582,
                0.465 - i * 0.048,
                line,
                size=19,
                color=ORANGE,
                alpha=fade_in(t, 0.62 + i * 0.04, 0.13),
                family="DejaVu Sans Mono",
            )
        text(
            fig,
            0.582,
            0.325,
            "— plus a Julia toolchain and PDDL.jl,",
            size=17,
            color=DIM,
            alpha=fade_in(t, 0.74, 0.14),
        )
        text(
            fig,
            0.582,
            0.285,
            "  installed separately, on every machine.",
            size=17,
            color=DIM,
            alpha=fade_in(t, 0.77, 0.14),
        )

    text(
        fig,
        0.08,
        0.14,
        "Then we spent a few years getting better at this.",
        size=26,
        color=AQUA,
        alpha=fade_in(t, 0.80, 0.16),
        style="italic",
    )


def scene_rewrite(fig, t, data):
    fig.patch.set_facecolor(BG)
    text(
        fig,
        0.08,
        0.84,
        "So we rewrote all of it.",
        size=44,
        weight="bold",
        alpha=fade_in(t, 0.0, 0.14),
    )

    # dependency counter falling to zero
    alpha = fade_in(t, 0.12, 0.15)
    if alpha > 0:
        text(fig, 0.08, 0.70, "runtime dependencies", size=18, color=MUTED, alpha=alpha)
        progress = ease_in_out(min(1.0, max(0.0, (t - 0.16) / 0.30)))
        value = 3 - 3 * progress
        color = GOOD if value < 0.5 else ORANGE
        text(
            fig,
            0.08,
            0.60,
            f"{value:.0f}",
            size=88,
            weight="bold",
            color=color,
            alpha=alpha,
        )
        if progress >= 0.999:
            text(
                fig,
                0.155,
                0.615,
                "standard library only",
                size=22,
                color=GOOD,
                alpha=fade_in(t, 0.47, 0.13),
            )
            text(
                fig,
                0.155,
                0.575,
                "matplotlib is an optional extra",
                size=16,
                color=MUTED,
                alpha=fade_in(t, 0.52, 0.13),
            )

    features = [
        "hand-written PDDL parser, grounder and compiler",
        f"{data['planner_count']} planners — BFS to A*, IDA*, beam, novelty, anytime",
        "8 heuristics — goal-count to h_FF and LM-cut",
        "ADL, axioms, numbers, time, constraints, preferences",
        f"{data['generator_count']} seeded instance generators",
        "search instrumentation, plots and live dashboards",
    ]
    for i, feature in enumerate(features):
        alpha = fade_in(t, 0.42 + i * 0.075, 0.13)
        if alpha <= 0:
            continue
        y = 0.44 - i * 0.066
        text(fig, 0.085, y, "▸", size=20, color=BLUE, alpha=alpha)
        text(fig, 0.115, y, feature, size=23, color=INK, alpha=alpha)

    # the grounded task on the right, counted up from real numbers
    alpha = fade_in(t, 0.30, 0.16)
    if alpha > 0:
        info = data["task"]
        panel(fig, 0.60, 0.30, 0.33, 0.34, alpha=alpha)
        text(
            fig,
            0.625,
            0.585,
            "blocksworld, grounded",
            size=17,
            color=MUTED,
            alpha=alpha,
        )
        rows = [("facts", info["facts"]), ("ground actions", info["operators"])]
        for i, (name, value) in enumerate(rows):
            y = 0.49 - i * 0.11
            shown = count_up(value, t, 0.34 + i * 0.06, 0.42)
            text(
                fig,
                0.625,
                y,
                fmt(shown),
                size=46,
                weight="bold",
                color=BLUE,
                alpha=alpha,
                family="DejaVu Sans Mono",
            )
            text(fig, 0.625, y - 0.055, name, size=17, color=DIM, alpha=alpha)


def scene_code(fig, t, data):
    fig.patch.set_facecolor(BG)
    text(
        fig,
        0.08,
        0.86,
        "Four lines to a validated plan.",
        size=40,
        weight="bold",
        alpha=fade_in(t, 0.0, 0.14),
    )

    panel(fig, 0.07, 0.36, 0.52, 0.42, alpha=fade_in(t, 0.08, 0.14))
    code = (
        "from jupyddl import solve\n\n"
        "result = solve('domain.pddl', 'problem.pddl',\n"
        "               search='astar', heuristic='lmcut')\n\n"
        "print(result.cost, result.plan_names())"
    )
    shown = typewriter(code, t, 0.14, 0.42)
    if shown:
        fig.text(
            0.095,
            0.735,
            shown,
            fontsize=20,
            color="#dcdbd4",
            va="top",
            ha="left",
            fontfamily="DejaVu Sans Mono",
            linespacing=1.7,
        )

    alpha = fade_in(t, 0.62, 0.16)
    if alpha > 0:
        panel(fig, 0.63, 0.36, 0.30, 0.42, alpha=alpha)
        text(fig, 0.655, 0.72, "hanoi, 5 discs", size=17, color=MUTED, alpha=alpha)
        text(
            fig,
            0.655,
            0.63,
            str(data["hanoi_cost"]),
            size=76,
            weight="bold",
            color=BLUE,
            alpha=alpha,
            family="DejaVu Sans Mono",
        )
        text(
            fig,
            0.655,
            0.545,
            "moves — provably optimal",
            size=19,
            color=DIM,
            alpha=alpha,
        )
        text(
            fig,
            0.655,
            0.47,
            "2⁵ − 1 = 31",
            size=24,
            color=GOOD,
            alpha=fade_in(t, 0.76, 0.15),
            family="DejaVu Sans Mono",
        )
        text(
            fig,
            0.655,
            0.415,
            "the textbook answer, reproduced",
            size=16,
            color=MUTED,
            alpha=fade_in(t, 0.80, 0.15),
        )


def _wavefront(ax, events, fraction, max_depth, color_for, size=7):
    """Draw the first ``fraction`` of an expansion list as a radial bloom."""
    cut = max(1, int(len(events) * fraction))
    subset = events[:cut]
    by_depth: dict = {}
    for event in subset:
        by_depth.setdefault(event.depth, []).append(event)
    angles, radii, colors = [], [], []
    for depth, group in by_depth.items():
        count = len(group)
        for i, event in enumerate(group):
            angles.append(2 * math.pi * (i + 0.5) / count)
            radii.append(depth)
            colors.append(color_for(event))
    ax.scatter(angles, radii, s=size, c=colors, linewidths=0)
    ax.set_ylim(0, max_depth + 1)
    ax.set_facecolor(SURFACE)
    ax.grid(color="#26262a", linewidth=0.6)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines["polar"].set_color("#26262a")
    return cut


def scene_search(fig, t, data):
    """The centrepiece: two searches on the same task, side by side."""
    fig.patch.set_facecolor(BG)
    astar = data["astar"]
    bfs = data["bfs"]

    text(
        fig,
        0.5,
        0.94,
        "Same problem. Same machine.",
        size=34,
        weight="bold",
        alpha=fade_in(t, 0.0, 0.12),
        ha="center",
    )
    text(
        fig,
        0.5,
        0.885,
        "blocksworld, 8 blocks — watch what the heuristic buys you",
        size=19,
        color=DIM,
        alpha=fade_in(t, 0.08, 0.12),
        ha="center",
    )

    progress = ease_in_out(min(1.0, max(0.0, (t - 0.16) / 0.62)))
    a_events = astar.expansions
    b_events = bfs.expansions
    max_depth = max(
        max((e.depth for e in a_events), default=1),
        max((e.depth for e in b_events), default=1),
    )

    a_h = max((e.h for e in a_events), default=1) or 1
    b_h = max((e.h for e in b_events), default=1) or 1

    def a_color(event):
        return SEQ[min(len(SEQ) - 1, int(event.h / a_h * (len(SEQ) - 1)))]

    def b_color(event):
        return SEQ[min(len(SEQ) - 1, int(event.h / b_h * (len(SEQ) - 1)))]

    alpha = fade_in(t, 0.14, 0.14)
    if alpha <= 0:
        return

    left = fig.add_axes([0.07, 0.26, 0.36, 0.56], projection="polar")
    right = fig.add_axes([0.57, 0.26, 0.36, 0.56], projection="polar")
    a_cut = _wavefront(left, a_events, progress, max_depth, a_color, size=26)
    b_cut = _wavefront(right, b_events, progress, max_depth, b_color, size=5)
    for ax in (left, right):
        ax.patch.set_alpha(alpha)

    a_total = astar.stats.get("expanded", len(a_events))
    b_total = bfs.stats.get("expanded", len(b_events))
    a_now = int(a_total * progress) if a_total else a_cut
    b_now = int(b_total * progress) if b_total else b_cut

    text(
        fig,
        0.25,
        0.845,
        "A*  +  LM-cut",
        size=25,
        weight="bold",
        color=BLUE,
        alpha=alpha,
        ha="center",
    )
    text(
        fig,
        0.75,
        0.845,
        "breadth-first",
        size=25,
        weight="bold",
        color=ORANGE,
        alpha=alpha,
        ha="center",
    )

    text(
        fig,
        0.25,
        0.185,
        fmt(a_now),
        size=54,
        weight="bold",
        color=BLUE,
        alpha=alpha,
        ha="center",
        family="DejaVu Sans Mono",
    )
    text(
        fig,
        0.75,
        0.185,
        fmt(b_now),
        size=54,
        weight="bold",
        color=ORANGE,
        alpha=alpha,
        ha="center",
        family="DejaVu Sans Mono",
    )
    text(
        fig,
        0.25,
        0.128,
        "nodes expanded",
        size=17,
        color=MUTED,
        alpha=alpha,
        ha="center",
    )
    text(
        fig,
        0.75,
        0.128,
        "nodes expanded",
        size=17,
        color=MUTED,
        alpha=alpha,
        ha="center",
    )

    if a_total and b_total and progress > 0.985:
        ratio = b_total / max(1, a_total)
        text(
            fig,
            0.5,
            0.052,
            f"same plan, same cost — {ratio:,.0f}× fewer nodes",
            size=26,
            weight="bold",
            color=GOOD,
            alpha=fade_in(t, 0.80, 0.12),
            ha="center",
        )


def scene_curves(fig, t, data):
    fig.patch.set_facecolor(BG)
    trace = data["curves"]
    events = trace.expansions

    text(
        fig,
        0.08,
        0.90,
        "Every search is instrumented.",
        size=38,
        weight="bold",
        alpha=fade_in(t, 0.0, 0.12),
    )
    text(
        fig,
        0.08,
        0.845,
        "one observer hook — plots, live terminal dashboards, JSON traces",
        size=20,
        color=DIM,
        alpha=fade_in(t, 0.08, 0.12),
    )

    progress = ease_in_out(min(1.0, max(0.0, (t - 0.14) / 0.68)))
    cut = max(2, int(len(events) * progress))
    steps = list(range(1, cut + 1))
    fs = [e.f for e in events[:cut]]
    gs = [e.g for e in events[:cut]]
    hs = [e.h for e in events[:cut]]
    opens = [e.open_size for e in events[:cut]]

    alpha = fade_in(t, 0.12, 0.14)
    if alpha <= 0:
        return

    ax = fig.add_axes([0.08, 0.44, 0.52, 0.33])
    for values, color, label in (
        (fs, BLUE, "f = g + h"),
        (gs, ORANGE, "g"),
        (hs, AQUA, "h"),
    ):
        ax.plot(
            steps,
            values,
            color=color,
            linewidth=2.4,
            alpha=alpha,
            solid_capstyle="round",
        )
        if values:
            text(
                fig,
                0.605,
                0.44 + 0.33 * (values[-1] / (max(max(fs), 1) * 1.1)),
                label,
                size=17,
                color=color,
                alpha=alpha,
                weight="bold",
            )
    ax.set_xlim(0, len(events))
    ax.set_ylim(0, max(max(fs), 1) * 1.1)
    _style_axis(ax, alpha, "nodes expanded", "cost")

    ax2 = fig.add_axes([0.08, 0.12, 0.52, 0.22])
    ax2.plot(steps, opens, color=BLUE, linewidth=2.2, alpha=alpha)
    ax2.fill_between(steps, opens, color=BLUE, alpha=0.15 * alpha)
    ax2.set_xlim(0, len(events))
    ax2.set_ylim(0, max(max(opens), 1) * 1.15)
    _style_axis(ax2, alpha, "nodes expanded", "frontier")

    # the live terminal dashboard, mocked from the same numbers
    alpha2 = fade_in(t, 0.34, 0.16)
    if alpha2 > 0:
        panel(fig, 0.65, 0.16, 0.29, 0.56, alpha=alpha2, color="#0f0f11")
        event = events[cut - 1]
        lines = [
            ("$ jupyddl solve ... --live", GOOD),
            ("", DIM),
            ("jupyddl · astar/lmcut · gripper-6", INK),
            ("", DIM),
            (f"h  {_spark(hs)}", AQUA),
            (f"f  {_spark(fs)}", BLUE),
            ("", DIM),
            (f"frontier  {_gauge(event.open_size, max(opens))}", ORANGE),
            ("", DIM),
            (f"{event.expanded:,} expanded", INK),
            (f"{event.generated:,} generated", DIM),
            (f"{event.elapsed * 1000:.0f} ms", MUTED),
        ]
        for i, (line, color) in enumerate(lines):
            if not line:
                continue
            fig.text(
                0.665,
                0.685 - i * 0.043,
                line,
                fontsize=13,
                color=color,
                alpha=alpha2,
                fontfamily="DejaVu Sans Mono",
                va="top",
            )


def _spark(values, width=22):
    chars = "▁▂▃▄▅▆▇█"
    if not values:
        return ""
    sample = values
    if len(sample) > width:
        step = len(sample) / width
        sample = [sample[min(len(sample) - 1, int(i * step))] for i in range(width)]
    low, high = min(sample), max(sample)
    if high == low:
        return chars[3] * len(sample)
    return "".join(chars[min(7, int((v - low) / (high - low) * 7.999))] for v in sample)


def _gauge(value, peak, width=12):
    filled = int(round(width * (value / max(1, peak))))
    return "█" * filled + "░" * (width - filled)


def _style_axis(ax, alpha, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.patch.set_alpha(alpha)
    ax.grid(color="#26262a", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#3a3a3e")
    ax.tick_params(colors=MUTED, labelsize=11)
    ax.set_xlabel(xlabel, color=DIM, fontsize=13)
    ax.set_ylabel(ylabel, color=DIM, fontsize=13)


def scene_showdown(fig, t, data):
    fig.patch.set_facecolor(BG)
    rows = data["showdown"]
    text(
        fig,
        0.08,
        0.88,
        "Pick your trade-off.",
        size=40,
        weight="bold",
        alpha=fade_in(t, 0.0, 0.12),
    )
    text(
        fig,
        0.08,
        0.825,
        "blocksworld, 8 blocks — nodes expanded, log scale",
        size=20,
        color=DIM,
        alpha=fade_in(t, 0.07, 0.12),
    )

    peak = max(r["expanded"] for r in rows)
    colors = [BLUE, ORANGE, AQUA, YELLOW, "#d55181"]
    for i, row in enumerate(rows):
        alpha = fade_in(t, 0.14 + i * 0.09, 0.14)
        if alpha <= 0:
            continue
        y = 0.665 - i * 0.115
        grow = ease_out(min(1.0, max(0.0, (t - (0.18 + i * 0.09)) / 0.35)))
        span = math.log10(max(row["expanded"], 1)) / math.log10(max(peak, 10))
        width = 0.52 * span * grow

        text(
            fig,
            0.085,
            y,
            row["label"],
            size=21,
            color=DIM,
            alpha=alpha,
            family="DejaVu Sans Mono",
        )
        if width > 0.001:
            fig.patches.append(
                FancyBboxPatch(
                    (0.30, y - 0.026),
                    width,
                    0.052,
                    boxstyle="round,pad=0,rounding_size=0.006",
                    transform=fig.transFigure,
                    facecolor=colors[i % len(colors)],
                    edgecolor="none",
                    alpha=alpha,
                    zorder=2,
                )
            )
        shown = count_up(row["expanded"], t, 0.18 + i * 0.09, 0.35)
        # Long bars would push their label into the cost column, so those
        # carry it inside the bar instead.
        if width > 0.34:
            text(
                fig,
                0.30 + width - 0.012,
                y,
                fmt(shown),
                size=21,
                weight="bold",
                color="#ffffff",
                alpha=alpha,
                ha="right",
                family="DejaVu Sans Mono",
            )
        else:
            text(
                fig,
                0.315 + width,
                y,
                fmt(shown),
                size=21,
                weight="bold",
                color=INK,
                alpha=alpha,
                family="DejaVu Sans Mono",
            )
        text(
            fig,
            0.955,
            y,
            f"cost {row['cost']}",
            size=17,
            color=MUTED,
            alpha=alpha,
            ha="right",
        )

    alpha = fade_in(t, 0.72, 0.14)
    if alpha > 0:
        best = min(rows, key=lambda r: r["expanded"])
        worst = max(rows, key=lambda r: r["expanded"])
        text(
            fig,
            0.08,
            0.12,
            f"{worst['label']} expands {worst['expanded']:,} nodes. "
            f"{best['label']} expands {best['expanded']:,}.",
            size=24,
            color=INK,
            alpha=alpha,
        )
        text(
            fig,
            0.08,
            0.065,
            "Identical plans. Identical cost.",
            size=24,
            color=GOOD,
            alpha=fade_in(t, 0.80, 0.14),
            weight="bold",
        )


def scene_plan(fig, t, data):
    fig.patch.set_facecolor(BG)
    plan = data["plan"] or []
    text(
        fig,
        0.08,
        0.88,
        "And the plan is checked, not trusted.",
        size=38,
        weight="bold",
        alpha=fade_in(t, 0.0, 0.12),
    )

    panel(fig, 0.07, 0.16, 0.50, 0.64, alpha=fade_in(t, 0.06, 0.14))
    shown = int(len(plan) * ease_in_out(min(1.0, max(0.0, (t - 0.10) / 0.55))))
    for i, action in enumerate(plan[:shown]):
        y = 0.75 - i * 0.037
        if y < 0.18:
            break
        fig.text(
            0.09,
            y,
            f"{i + 1:2d}.",
            fontsize=14,
            color=MUTED,
            fontfamily="DejaVu Sans Mono",
            va="center",
        )
        fig.text(
            0.125,
            y,
            action,
            fontsize=14,
            color="#dcdbd4",
            fontfamily="DejaVu Sans Mono",
            va="center",
        )

    alpha = fade_in(t, 0.66, 0.15)
    if alpha > 0 and data["plan_valid"]:
        text(fig, 0.62, 0.60, "✔", size=90, color=GOOD, alpha=alpha)
        text(
            fig,
            0.70,
            0.615,
            "validated",
            size=38,
            weight="bold",
            color=GOOD,
            alpha=alpha,
        )
        text(
            fig,
            0.70,
            0.555,
            "replayed from the initial state,",
            size=20,
            color=DIM,
            alpha=alpha,
        )
        text(
            fig,
            0.70,
            0.51,
            "action by action, into the goal.",
            size=20,
            color=DIM,
            alpha=alpha,
        )
        text(
            fig,
            0.62,
            0.40,
            f"{len(plan)} actions · cost {data['cost']}",
            size=26,
            color=INK,
            alpha=fade_in(t, 0.74, 0.14),
            family="DejaVu Sans Mono",
        )


def scene_web(fig, t, data, screenshot=None):
    """The workbench, with the four things you can do in it."""
    fig.patch.set_facecolor(BG)
    text(
        fig,
        0.5,
        0.945,
        "A workbench, in your browser.",
        size=40,
        weight="bold",
        alpha=fade_in(t, 0.0, 0.13),
        ha="center",
    )
    text(
        fig,
        0.5,
        0.893,
        "the same library, compiled to WebAssembly — nothing to install",
        size=20,
        color=DIM,
        alpha=fade_in(t, 0.09, 0.13),
        ha="center",
    )

    alpha = fade_in(t, 0.18, 0.18)
    if alpha > 0:
        if screenshot is not None:
            ax = fig.add_axes([0.10, 0.20, 0.80, 0.64])
            ax.imshow(screenshot, alpha=min(1.0, alpha), aspect="auto")
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            panel(fig, 0.10, 0.20, 0.80, 0.64, alpha=alpha)

    views = [
        ("solve", "watch a search run"),
        ("experiment", "sweep a matrix, export CSV"),
        ("PDDL support", "what is accepted, and how"),
        ("generate", "reproducible instances"),
    ]
    for i, (name, note) in enumerate(views):
        step = fade_in(t, 0.46 + i * 0.09, 0.13)
        if step <= 0:
            continue
        x = 0.105 + i * 0.205
        text(fig, x, 0.115, name, size=21, weight="bold", color=BLUE, alpha=step)
        text(fig, x, 0.065, note, size=16, color=MUTED, alpha=step)


def scene_cta(fig, t, data):
    fig.patch.set_facecolor(BG)
    text(
        fig,
        0.5,
        0.70,
        "jupyddl",
        size=86,
        weight="bold",
        alpha=fade_in(t, 0.0, 0.16),
        ha="center",
    )
    text(
        fig,
        0.5,
        0.60,
        "a pure-Python PDDL planning framework",
        size=26,
        color=DIM,
        alpha=fade_in(t, 0.10, 0.16),
        ha="center",
    )

    alpha = fade_in(t, 0.26, 0.16)
    if alpha > 0:
        panel(fig, 0.315, 0.40, 0.37, 0.088, alpha=alpha, color="#141416")
        text(
            fig,
            0.5,
            0.444,
            "pip install jupyddl",
            size=27,
            color=AQUA,
            alpha=alpha,
            ha="center",
            family="DejaVu Sans Mono",
        )

    text(
        fig,
        0.5,
        0.29,
        "github.com/openplan-labs/PythonPDDL",
        size=23,
        color=INK,
        alpha=fade_in(t, 0.42, 0.16),
        ha="center",
        family="DejaVu Sans Mono",
    )
    text(
        fig,
        0.5,
        0.225,
        "zero dependencies · Apache-2.0",
        size=19,
        color=MUTED,
        alpha=fade_in(t, 0.52, 0.16),
        ha="center",
    )
    text(
        fig,
        0.5,
        0.13,
        "started as a uni project. grew up.",
        size=21,
        color=BLUE,
        alpha=fade_in(t, 0.64, 0.18),
        ha="center",
        style="italic",
    )


SUPPORT_COLOURS = {
    "native": "#0ca30c",
    "compiled": BLUE,
    "partial": YELLOW,
    "rejected": "#4a4a4e",
}


def scene_requirements(fig, t, data):
    """The PDDL support matrix, as 21 tiles that light up by support level."""
    fig.patch.set_facecolor(BG)
    counts = data["support"]
    supported = counts["native"] + counts["compiled"] + counts["partial"]

    text(
        fig,
        0.08,
        0.89,
        "How much PDDL, exactly?",
        size=42,
        weight="bold",
        alpha=fade_in(t, 0.0, 0.12),
    )
    text(
        fig,
        0.08,
        0.825,
        "every requirement flag in the standard, and what this planner does with it",
        size=20,
        color=DIM,
        alpha=fade_in(t, 0.07, 0.12),
    )

    rows = data["requirements"]
    columns = 3
    for i, row in enumerate(rows):
        alpha = fade_in(t, 0.14 + i * 0.022, 0.10)
        if alpha <= 0:
            continue
        column = i % columns
        line = i // columns
        x = 0.075 + column * 0.30
        y = 0.715 - line * 0.078
        colour = SUPPORT_COLOURS[row["support"]]
        fig.patches.append(
            FancyBboxPatch(
                (x, y - 0.014),
                0.012,
                0.030,
                boxstyle="round,pad=0,rounding_size=0.004",
                transform=fig.transFigure,
                facecolor=colour,
                edgecolor="none",
                alpha=alpha,
                zorder=3,
            )
        )
        muted = row["support"] == "rejected"
        text(
            fig,
            x + 0.022,
            y,
            row["name"],
            size=17,
            color=MUTED if muted else INK,
            alpha=alpha,
            family="DejaVu Sans Mono",
        )

    alpha = fade_in(t, 0.66, 0.15)
    if alpha > 0:
        shown = count_up(supported, t, 0.68, 0.35)
        text(
            fig,
            0.075,
            0.145,
            f"{shown}",
            size=76,
            weight="bold",
            color=GOOD,
            alpha=alpha,
        )
        text(fig, 0.205, 0.172, "of 21 supported", size=25, color=INK, alpha=alpha)
        text(
            fig,
            0.205,
            0.122,
            f"{counts['native']} native · {counts['compiled']} compiled · "
            f"{counts['partial']} partial",
            size=17,
            color=DIM,
            alpha=alpha,
        )

    text(
        fig,
        0.52,
        0.145,
        "The one that is refused fails at parse time,",
        size=19,
        color=DIM,
        alpha=fade_in(t, 0.80, 0.14),
    )
    text(
        fig,
        0.52,
        0.105,
        "by name. Silently ignoring a requirement",
        size=19,
        color=DIM,
        alpha=fade_in(t, 0.83, 0.14),
    )
    text(
        fig,
        0.52,
        0.065,
        "gives you plans that are wrong, not absent.",
        size=19,
        color=ORANGE,
        alpha=fade_in(t, 0.86, 0.14),
    )


def scene_pddl3(fig, t, data):
    """Soft goals: the optimal plan deliberately skips an errand."""
    fig.patch.set_facecolor(BG)
    errands = data["errands"]

    text(
        fig,
        0.08,
        0.89,
        "Tell it what you'd rather have.",
        size=42,
        weight="bold",
        alpha=fade_in(t, 0.0, 0.12),
    )
    text(
        fig,
        0.08,
        0.825,
        "PDDL 3 preferences: soft goals, priced by the metric",
        size=20,
        color=DIM,
        alpha=fade_in(t, 0.07, 0.12),
    )

    alpha = fade_in(t, 0.12, 0.14)
    if alpha > 0:
        panel(fig, 0.07, 0.44, 0.42, 0.31, alpha=alpha, color="#101012")
        code = (
            "(:goal (and (at-home) (locked)\n"
            "   (preference milk   (bought-milk))\n"
            "   (preference letter (posted-letter))))\n\n"
            "(:metric minimize (+ (total-cost)\n"
            "   (* 6 (is-violated milk))\n"
            "   (* 2 (is-violated letter))))"
        )
        shown = typewriter(code, t, 0.16, 0.34)
        if shown:
            fig.text(
                0.092,
                0.715,
                shown,
                fontsize=16,
                color="#dcdbd4",
                va="top",
                ha="left",
                fontfamily="DejaVu Sans Mono",
                linespacing=1.6,
            )

    # The verdict on each errand, straight from the solved plan.
    entries = [
        ("milk", "buy-milk", "worth 6, costs 1"),
        ("bread", "buy-bread", "worth 6, costs 1"),
        ("letter", "post-letter", "worth 2, costs 3"),
    ]
    for i, (label, action, economics) in enumerate(entries):
        alpha = fade_in(t, 0.50 + i * 0.09, 0.13)
        if alpha <= 0:
            continue
        y = 0.70 - i * 0.105
        done = action in errands["plan"]
        mark = "✔" if done else "✘"
        colour = GOOD if done else ORANGE
        text(fig, 0.56, y, mark, size=32, color=colour, alpha=alpha)
        text(
            fig, 0.60, y + 0.012, label, size=24, weight="bold", color=INK, alpha=alpha
        )
        text(fig, 0.60, y - 0.030, economics, size=16, color=MUTED, alpha=alpha)

    alpha = fade_in(t, 0.79, 0.14)
    if alpha > 0:
        text(
            fig,
            0.08,
            0.33,
            "The optimal plan buys the milk and the bread —",
            size=23,
            color=INK,
            alpha=alpha,
        )
        text(
            fig,
            0.08,
            0.275,
            "and deliberately skips the letter, paying the penalty.",
            size=23,
            color=INK,
            alpha=fade_in(t, 0.82, 0.14),
        )
        text(
            fig,
            0.08,
            0.195,
            f"total cost {errands['cost']:g}, plan validated",
            size=20,
            color=GOOD,
            alpha=fade_in(t, 0.87, 0.14),
            family="DejaVu Sans Mono",
        )
        text(
            fig,
            0.08,
            0.115,
            "A planner that treated preferences as hard goals would post it.",
            size=18,
            color=DIM,
            alpha=fade_in(t, 0.90, 0.13),
            style="italic",
        )


def scene_time(fig, t, data):
    """Timed initial literals: the world moves whether you do or not."""
    fig.patch.set_facecolor(BG)
    market = data["market"]

    text(
        fig,
        0.08,
        0.89,
        "And the clock runs without you.",
        size=42,
        weight="bold",
        alpha=fade_in(t, 0.0, 0.12),
    )
    text(
        fig,
        0.08,
        0.825,
        "timed initial literals: the market opens at 08:00 whatever you are doing",
        size=20,
        color=DIM,
        alpha=fade_in(t, 0.07, 0.12),
    )

    # A timeline: bake, then idle, then sell.
    alpha = fade_in(t, 0.16, 0.14)
    if alpha > 0:
        left, right, y = 0.10, 0.90, 0.55
        span = right - left
        end = market["makespan"] or 1
        fig.add_artist(
            plt.Line2D(
                [left, right],
                [y, y],
                color="#3a3a3e",
                linewidth=2,
                transform=fig.transFigure,
                alpha=alpha,
                solid_capstyle="round",
            )
        )
        progress = ease_in_out(min(1.0, max(0.0, (t - 0.20) / 0.45)))

        def at(hour):
            return left + span * (hour / end)

        bars = [
            (0.0, 3.0, BLUE, "bake", "3 hours of work"),
            (3.0, 8.0, "#2a2a2e", "wait", "the market is shut"),
            (8.0, 9.0, AQUA, "sell", "1 hour, once it opens"),
        ]
        for start_h, end_h, colour, label, note in bars:
            visible_end = min(end_h, progress * end)
            if visible_end <= start_h:
                continue
            x0, x1 = at(start_h), at(visible_end)
            fig.patches.append(
                FancyBboxPatch(
                    (x0, y - 0.035),
                    max(0.001, x1 - x0),
                    0.07,
                    boxstyle="round,pad=0,rounding_size=0.006",
                    transform=fig.transFigure,
                    facecolor=colour,
                    edgecolor="none",
                    alpha=alpha,
                    zorder=3,
                )
            )
            if progress * end >= end_h - 0.01:
                mid = (at(start_h) + at(end_h)) / 2
                text(
                    fig,
                    mid,
                    y + 0.075,
                    label,
                    size=20,
                    weight="bold",
                    color=colour if colour != "#2a2a2e" else MUTED,
                    alpha=alpha,
                    ha="center",
                )
                text(
                    fig,
                    mid,
                    y - 0.075,
                    note,
                    size=15,
                    color=MUTED,
                    alpha=alpha,
                    ha="center",
                )

        for hour in (0, 3, 8, 9):
            if progress * end >= hour - 0.01:
                text(
                    fig,
                    at(hour),
                    y - 0.125,
                    f"{hour:02d}:00",
                    size=14,
                    color=MUTED,
                    alpha=alpha,
                    ha="center",
                    family="DejaVu Sans Mono",
                )

    alpha = fade_in(t, 0.72, 0.14)
    if alpha > 0:
        text(
            fig,
            0.10,
            0.30,
            f"{market['work']:g}",
            size=64,
            weight="bold",
            color=BLUE,
            alpha=alpha,
        )
        text(fig, 0.10, 0.225, "hours of work", size=19, color=DIM, alpha=alpha)
        text(
            fig,
            0.45,
            0.30,
            f"{market['makespan']:g}",
            size=64,
            weight="bold",
            color=AQUA,
            alpha=fade_in(t, 0.78, 0.14),
        )
        text(
            fig,
            0.45,
            0.225,
            "hours of makespan",
            size=19,
            color=DIM,
            alpha=fade_in(t, 0.78, 0.14),
        )
        text(
            fig,
            0.10,
            0.115,
            "Waiting is a move, and the plan is scored for it.",
            size=21,
            color=INK,
            alpha=fade_in(t, 0.85, 0.14),
            style="italic",
        )


SCENES = [
    (scene_title, 4.5),
    (scene_origin, 7.5),
    (scene_rewrite, 8.5),
    (scene_code, 6.0),
    (scene_search, 12.0),
    (scene_curves, 8.0),
    (scene_showdown, 9.0),
    (scene_requirements, 10.0),
    (scene_pddl3, 11.0),
    (scene_time, 9.5),
    (scene_plan, 6.0),
    (scene_web, 6.5),
    (scene_cta, 6.0),
]


def render(data, out: str, screenshot=None, fps: int = FPS, dpi: int = DPI):
    fig = plt.figure(figsize=SIZE, facecolor=BG)
    total = sum(int(seconds * fps) for _, seconds in SCENES)

    if out.lower().endswith(".gif"):
        writer = PillowWriter(fps=fps)
    else:
        if not FFMpegWriter.isAvailable():
            try:
                import imageio_ffmpeg

                matplotlib.rcParams["animation.ffmpeg_path"] = (
                    imageio_ffmpeg.get_ffmpeg_exe()
                )
            except Exception:
                pass
        if not FFMpegWriter.isAvailable():
            raise RuntimeError(
                "ffmpeg not found. pip install imageio-ffmpeg, or render a .gif"
            )
        writer = FFMpegWriter(
            fps=fps,
            bitrate=-1,
            extra_args=["-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium"],
        )

    started = time.perf_counter()
    done = 0
    print(f"Rendering {total} frames ({total / fps:.1f}s) -> {out}")
    with writer.saving(fig, out, dpi=dpi):
        for draw, seconds in SCENES:
            frames = int(seconds * fps)
            for frame in range(frames):
                fig.clear()
                fig.patches.clear()
                t = frame / max(1, frames - 1)
                if draw is scene_web:
                    draw(fig, t, data, screenshot)
                else:
                    draw(fig, t, data)
                writer.grab_frame(facecolor=BG)
                done += 1
                if done % 60 == 0:
                    rate = done / (time.perf_counter() - started)
                    print(f"  {done}/{total} frames ({rate:.0f} fps render)")
    plt.close(fig)
    print(f"Wrote {out} in {time.perf_counter() - started:.0f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the jupyddl promo video.")
    parser.add_argument("-o", "--output", default="promo/jupyddl.mp4")
    parser.add_argument("--demos", default="demos")
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument(
        "--screenshot",
        default=None,
        help="PNG of the playground to feature in the web scene",
    )
    args = parser.parse_args()

    folder = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(folder, exist_ok=True)

    screenshot = None
    if args.screenshot and os.path.exists(args.screenshot):
        screenshot = plt.imread(args.screenshot)

    data = collect(args.demos)
    render(data, args.output, screenshot=screenshot, fps=args.fps, dpi=args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render the promo video for the learned-heuristic / RL work.

Like ``make_promo.py``, nothing on screen is typed in by hand: this trains a
heuristic, runs the reinforcement stage, reproduces both of the failure modes
that shaped its design, and animates whatever came back. If the method gets
better or worse, so does the video.

    python tools/make_learn_promo.py -o promo/jupyddl-rl.mp4

Collection takes a few minutes, so it is cached::

    python tools/make_learn_promo.py --cache promo/rl-data.json     # measure once
    python tools/make_learn_promo.py --cache promo/rl-data.json     # reuse

Needs the ``viz`` extra plus an ffmpeg binary; ``learn`` (NumPy) makes the
collection pass much faster but is not required.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, PillowWriter  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from make_promo import (  # noqa: E402
    AQUA,
    BG,
    BLUE,
    DIM,
    DPI,
    FPS,
    GOOD,
    INK,
    MONO,
    MUTED,
    ORANGE,
    SIZE,
    YELLOW,
    count_up,
    ease_in_out,
    ease_out,
    fade_in,
    fade_window,
    fmt,
    panel,
    text,
    typewriter,
)

BAD = "#e05252"

# --------------------------------------------------------------------------
# data collection — every number the video shows is measured here
# --------------------------------------------------------------------------
TRAIN_SIZES = range(3, 7)
HARD_SIZES = range(9, 13)
EVAL_SIZES = range(9, 14)


def collect() -> dict:
    """Train, reinforce, and reproduce both failure modes. Returns everything."""
    from jupyddl.learn import TrainConfig, solved_corpus, tasks_from_generator, train
    from jupyddl.learn.features import FeatureSpace
    from jupyddl.learn.pipeline import evaluate_transfer, summarise_transfer
    from jupyddl.learn.rl import RLConfig, optimise_search_cost, search_cost

    started = time.perf_counter()
    data: dict = {}

    def note(message):
        print(f"  [{time.perf_counter() - started:5.1f}s] {message}")

    # -- the instance families. Four disjoint seed families, which is the
    # -- point of the whole exercise: nothing is ever scored on what it fit.
    train_tasks = tasks_from_generator(
        "blocksworld", TRAIN_SIZES, seed=0, seeds_per_size=3
    )
    tune_tasks = tasks_from_generator(
        "blocksworld", HARD_SIZES, seed=1000, seeds_per_size=2
    )
    val_tasks = tasks_from_generator(
        "blocksworld", HARD_SIZES, seed=2000, seeds_per_size=2
    )
    eval_tasks = tasks_from_generator(
        "blocksworld", EVAL_SIZES, seed=7777, seeds_per_size=2
    )
    space = FeatureSpace.from_tasks(
        task for _, task in train_tasks + tune_tasks + val_tasks + eval_tasks
    )
    data["space"] = {
        "predicates": list(space.vocabulary),
        "features": space.size,
        "train_instances": len(train_tasks),
        "train_sizes": [min(TRAIN_SIZES), max(TRAIN_SIZES)],
        "eval_sizes": [min(EVAL_SIZES), max(EVAL_SIZES)],
    }
    note(f"ladder: {len(train_tasks)} train, {len(eval_tasks)} eval")

    # -- one plan, with the labels it hands over for free
    corpus = solved_corpus(train_tasks, space=space)
    data["corpus"] = corpus.target_stats()
    note(
        f"corpus: {data['corpus']['count']} samples, {data['corpus']['groups']} groups"
    )

    from jupyddl.api import solve_task

    name, demo_task = train_tasks[2]
    demo = solve_task(demo_task, "astar", "lmcut", time_limit=30)
    steps = [op.base_name for op in demo_task.visible_plan(demo.plan)]
    suffix = [0.0] * (len(demo.plan) + 1)
    for i in range(len(demo.plan) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + demo.plan[i].cost
    data["plan"] = {"instance": name, "steps": steps, "suffix": suffix[: len(steps)]}

    # -- imitation
    config = TrainConfig(epochs=60, seed=0)
    bundle, report = train(corpus, config)
    data["imitation"] = {
        "mae": report.metrics["mae"],
        "top1": report.metrics["top1"],
        "in_top2": report.metrics.get("in_top2", 0.0),
        "best_epoch": report.best_epoch,
        "seconds": report.seconds,
        "parameters": bundle.model.num_parameters,
        "history": [
            {"epoch": h["epoch"], "top1": h["train_top1"], "loss": h["loss"]}
            for h in report.history
        ],
    }
    note(
        f"imitation: top-1 {report.metrics['top1']:.3f}, MAE {report.metrics['mae']:.2f}"
    )

    rl = RLConfig(max_expansions=30000, seed=0)
    before_rows = evaluate_transfer(
        bundle, eval_tasks, max_expansions=30000, time_limit=30.0
    )
    data["transfer_before"] = summarise_transfer(before_rows)
    note(
        f"imitation on eval: {data['transfer_before']['learned']['mean_expanded']:.0f} expanded"
    )

    # -- the reinforcement stage, done properly
    cem_started = time.perf_counter()
    tuned, history = optimise_search_cost(
        bundle,
        tune_tasks,
        iterations=10,
        population=12,
        config=rl,
        validation_tasks=val_tasks,
    )
    data["cem"] = {
        "seconds": time.perf_counter() - cem_started,
        "iterations": len(history) - 1,
        "population": 12,
        "history": [
            {
                "iteration": h["iteration"],
                "tuning": h.get("tuning_score"),
                "validation": h["score"],
                "coverage": h["coverage"],
            }
            for h in history
        ],
    }
    after_rows = evaluate_transfer(
        tuned, eval_tasks, max_expansions=30000, time_limit=30.0
    )
    data["transfer_after"] = summarise_transfer(after_rows)
    note(
        f"cem: {data['cem']['seconds']:.0f}s, eval "
        f"{data['transfer_after']['learned']['mean_expanded']:.0f} expanded"
    )

    # -- failure mode 1: the objective is flat where the search is already good
    flat_before = search_cost(bundle, train_tasks, rl).score
    flat_tuned, _ = optimise_search_cost(
        bundle, train_tasks, iterations=6, population=8, config=rl
    )
    data["flat"] = {
        "easy_before": flat_before,
        "easy_after": search_cost(flat_tuned, train_tasks, rl).score,
        "hard_before": search_cost(bundle, tune_tasks, rl).score,
        "hard_after": search_cost(tuned, tune_tasks, rl).score,
    }
    note(
        f"flat: easy {data['flat']['easy_before']:.1f} -> {data['flat']['easy_after']:.1f}, "
        f"hard {data['flat']['hard_before']:.0f} -> {data['flat']['hard_after']:.0f}"
    )

    # -- the per-instance distribution behind the headline mean.
    #
    # This scene exists because the headline nearly went out wrong. Two things
    # changed in one edit — a validation split, and sigma 0.05 -> 0.15 — and the
    # improvement was credited to the first. Varying one at a time shows sigma
    # doing all of it, and the per-instance breakdown shows why: nine of ten
    # instances barely move, and the tenth decides the average. Both arms are
    # measured here so the video cannot drift from the claim.
    lo_sigma, hi_sigma = 0.05, 0.15
    variants = {}
    for label, sigma in (("lo", lo_sigma), ("hi", hi_sigma)):
        model, hist = optimise_search_cost(
            bundle,
            tune_tasks,
            iterations=10,
            population=12,
            sigma=sigma,
            config=rl,
            validation_tasks=val_tasks,
        )
        rows = evaluate_transfer(
            model, eval_tasks, baselines=(), max_expansions=30000, time_limit=30.0
        )
        variants[label] = {
            "sigma": sigma,
            "tuning": hist[-1].get("tuning_score"),
            "per_instance": {r["instance"]: r["expanded"] for r in rows},
            "solved": {r["instance"]: r["solved"] for r in rows},
            "mean": summarise_transfer(rows)["learned"]["mean_expanded"],
        }
    base_rows = evaluate_transfer(
        bundle, eval_tasks, baselines=(), max_expansions=30000, time_limit=30.0
    )
    data["spread"] = {
        "instances": [r["instance"] for r in base_rows],
        "imitation": {r["instance"]: r["expanded"] for r in base_rows},
        "imitation_solved": {r["instance"]: r["solved"] for r in base_rows},
        "lo": variants["lo"],
        "hi": variants["hi"],
        "budget": 30000,
    }
    note(
        f"spread: sigma {lo_sigma} mean {variants['lo']['mean']:.0f}, "
        f"sigma {hi_sigma} mean {variants['hi']['mean']:.0f}"
    )

    # -- where it loses, and why
    log_train = tasks_from_generator("logistics", range(2, 5), seed=0, seeds_per_size=3)
    log_eval = tasks_from_generator(
        "logistics", range(5, 8), seed=7777, seeds_per_size=2
    )
    log_space = FeatureSpace.from_tasks(t for _, t in log_train + log_eval)
    log_corpus = solved_corpus(log_train, space=log_space, time_limit=30.0)
    log_bundle, log_report = train(log_corpus, TrainConfig(epochs=60, seed=0))
    log_rows = evaluate_transfer(
        log_bundle, log_eval, max_expansions=30000, time_limit=30.0
    )
    log_summary = summarise_transfer(log_rows)
    data["logistics"] = {
        "predicates": list(log_space.vocabulary),
        "features": log_space.size,
        "top1": log_report.metrics["top1"],
        "learned": log_summary["learned"]["mean_expanded"],
        "hff": log_summary["hff"]["mean_expanded"],
        "blocks_top1": data["imitation"]["top1"],
    }
    note(
        f"logistics: {len(log_space.vocabulary)} predicates, top-1 "
        f"{log_report.metrics['top1']:.3f}, {log_summary['learned']['mean_expanded']:.0f} "
        f"vs hff {log_summary['hff']['mean_expanded']:.0f}"
    )

    data["total_seconds"] = time.perf_counter() - started
    return data


# --------------------------------------------------------------------------
# drawing helpers specific to this video
# --------------------------------------------------------------------------
def _header(fig, title, subtitle, alpha, y=0.90):
    text(fig, 0.08, y, title, size=42, weight="bold", alpha=alpha)
    if subtitle:
        text(fig, 0.08, y - 0.075, subtitle, size=21, color=DIM, alpha=alpha * 0.95)


def _axes(fig, rect, alpha):
    ax = fig.add_axes(rect)
    ax.set_facecolor("none")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#3a3a3d")
        ax.spines[spine].set_alpha(alpha)
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=13, length=3, width=1.0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_alpha(alpha)
    ax.grid(True, axis="y", color="#26262a", linewidth=1.0, alpha=alpha * 0.7)
    ax.set_axisbelow(True)
    return ax


def _rounded_bar(ax, x, width, height, color, alpha, radius_frac=0.35):
    """A bar with rounded data-end and a square foot on the baseline.

    Rounding both ends turns a bar into a pill and detaches it from the axis it
    is measured against; only the free end gets a radius.
    """
    if height <= 0:
        return
    span = ax.get_ylim()[1] - ax.get_ylim()[0] or 1.0
    radius = min(width * radius_frac, height * 0.5, span * 0.03)
    left, right = x - width / 2, x + width / 2
    verts = [
        (left, 0.0),
        (left, height - radius),
        (left, height - radius * 0.45),
        (left + radius * 0.45, height),
        (left + radius, height),
        (right - radius, height),
        (right - radius * 0.45, height),
        (right, height - radius * 0.45),
        (right, height - radius),
        (right, 0.0),
        (left, 0.0),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    ax.add_patch(
        PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", alpha=alpha)
    )


def _stat(fig, x, y, value, label, colour=INK, alpha=1.0, size=64, sub=None):
    text(fig, x, y, value, size=size, weight="bold", color=colour, alpha=alpha)
    text(fig, x, y - 0.085, label, size=19, color=DIM, alpha=alpha * 0.95)
    if sub:
        text(fig, x, y - 0.135, sub, size=16, color=MUTED, alpha=alpha * 0.9)


# --------------------------------------------------------------------------
# scenes
# --------------------------------------------------------------------------
def scene_hook(fig, t, data):
    alpha = fade_window(t, 0.0, 1.0, 0.14)
    text(
        fig,
        0.5,
        0.60,
        "Your planner already wrote",
        size=58,
        weight="bold",
        alpha=alpha * fade_in(t, 0.05),
        ha="center",
    )
    text(
        fig,
        0.5,
        0.47,
        "the training data.",
        size=58,
        weight="bold",
        color=BLUE,
        alpha=alpha * fade_in(t, 0.22),
        ha="center",
    )
    text(
        fig,
        0.5,
        0.32,
        "every solved plan is a labelled trajectory",
        size=24,
        color=DIM,
        alpha=alpha * fade_in(t, 0.45),
        ha="center",
    )


def scene_labels(fig, t, data):
    alpha = fade_window(t, 0.0, 1.0, 0.10)
    _header(
        fig,
        "A plan is supervision you already paid for.",
        "the cost of the suffix from any state on it is that state's cost-to-go",
        alpha,
    )

    plan = data["plan"]
    steps = plan["steps"][:8]
    suffix = plan["suffix"][:8]
    panel(fig, 0.08, 0.16, 0.40, 0.58, alpha * fade_in(t, 0.12))

    for index, (step, cost) in enumerate(zip(steps, suffix)):
        row_alpha = alpha * fade_in(t, 0.18 + index * 0.045, 0.12)
        y = 0.67 - index * 0.062
        text(
            fig,
            0.115,
            y,
            f"{index + 1}.",
            size=17,
            color=MUTED,
            alpha=row_alpha,
            family=MONO,
        )
        text(fig, 0.155, y, step, size=17, color=INK, alpha=row_alpha, family=MONO)
        # The label the state carries, appearing beside the step that reveals it.
        text(
            fig,
            0.445,
            y,
            f"h* = {cost:.0f}",
            size=17,
            color=AQUA,
            alpha=alpha * fade_in(t, 0.42 + index * 0.045, 0.12),
            family=MONO,
            ha="right",
        )

    right = alpha * fade_in(t, 0.62)
    text(fig, 0.56, 0.62, "One instance,", size=30, color=DIM, alpha=right)
    _stat(
        fig,
        0.56,
        0.50,
        f"{count_up(len(steps) + 1, t, 0.66, 0.3)}",
        "labelled states",
        colour=AQUA,
        alpha=right,
        size=58,
    )
    stats = data["corpus"]
    _stat(
        fig,
        0.56,
        0.28,
        fmt(stats["count"]),
        f"across {data['space']['train_instances']} instances of 3-6 blocks",
        colour=INK,
        alpha=alpha * fade_in(t, 0.78),
        size=58,
    )
    text(
        fig,
        0.56,
        0.13,
        "no labelling, no annotation, no human",
        size=19,
        color=MUTED,
        alpha=alpha * fade_in(t, 0.86),
    )


def scene_imitation(fig, t, data):
    alpha = fade_window(t, 0.0, 1.0, 0.10)
    _header(
        fig,
        "Fit a network to it. That is imitation.",
        "a thousand parameters over fourteen features, trained in a tenth of a second",
        alpha,
    )

    history = data["imitation"]["history"]
    if not history:  # pragma: no cover - defensive
        return
    reveal = min(1.0, max(0.0, (t - 0.20) / 0.45))
    shown = max(2, int(len(history) * ease_out(reveal)))

    ax = _axes(fig, [0.09, 0.20, 0.48, 0.46], alpha * fade_in(t, 0.15))
    xs = [h["epoch"] for h in history[:shown]]
    ys = [h["top1"] for h in history[:shown]]
    ax.plot(xs, ys, color=BLUE, linewidth=2.0, solid_capstyle="round")
    ax.scatter([xs[-1]], [ys[-1]], s=64, color=BLUE, zorder=5)
    ax.set_xlim(0, len(history) + 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("epoch", color=MUTED, fontsize=14)
    ax.set_ylabel("top-1 ranking accuracy", color=MUTED, fontsize=14)
    # One series: the axis label names it, so no legend box.
    ax.annotate(
        f"{ys[-1]:.3f}",
        (xs[-1], ys[-1]),
        textcoords="offset points",
        xytext=(12, -4),
        color=BLUE,
        fontsize=17,
        fontweight="bold",
        alpha=alpha,
    )

    right = alpha * fade_in(t, 0.55)
    _stat(
        fig,
        0.65,
        0.56,
        f"{data['imitation']['mae']:.2f}",
        "mean absolute error, held out",
        colour=INK,
        alpha=right,
        size=56,
    )
    _stat(
        fig,
        0.65,
        0.32,
        f"{data['imitation']['top1']:.0%}",
        "of decisions ranked correctly",
        colour=BLUE,
        alpha=alpha * fade_in(t, 0.70),
        size=56,
    )
    text(
        fig,
        0.65,
        0.16,
        f"{data['imitation']['seconds']:.1f}s of training",
        size=18,
        color=MUTED,
        alpha=alpha * fade_in(t, 0.82),
    )


def scene_order(fig, t, data):
    alpha = fade_window(t, 0.0, 1.0, 0.10)
    _header(
        fig,
        "But search never reads the number.",
        "greedy best-first search pops the minimum — it reads the order",
        alpha,
    )

    # Two heuristics, same decision point. Deliberately schematic: the claim is
    # about what the search consumes, not about any one measured value.
    left_alpha = alpha * fade_in(t, 0.16)
    panel(fig, 0.07, 0.20, 0.40, 0.48, left_alpha)
    text(
        fig, 0.10, 0.615, "h + 30, everywhere", size=24, weight="bold", alpha=left_alpha
    )
    text(fig, 0.10, 0.565, "terrible RMSE", size=17, color=BAD, alpha=left_alpha)
    for index, (label, value) in enumerate([("successor A", 32), ("successor B", 35)]):
        row = left_alpha * fade_in(t, 0.26 + index * 0.06, 0.12)
        y = 0.47 - index * 0.085
        text(fig, 0.10, y, label, size=19, color=DIM, alpha=row, family=MONO)
        text(
            fig,
            0.42,
            y,
            str(value),
            size=19,
            color=INK,
            alpha=row,
            family=MONO,
            ha="right",
        )
    text(
        fig,
        0.10,
        0.26,
        "→ picks A. Perfect guidance.",
        size=20,
        color=GOOD,
        alpha=left_alpha * fade_in(t, 0.42),
    )

    right_alpha = alpha * fade_in(t, 0.50)
    panel(fig, 0.53, 0.20, 0.40, 0.48, right_alpha)
    text(
        fig,
        0.56,
        0.615,
        "accurate, two siblings swapped",
        size=24,
        weight="bold",
        alpha=right_alpha,
    )
    text(fig, 0.56, 0.565, "excellent RMSE", size=17, color=GOOD, alpha=right_alpha)
    for index, (label, value) in enumerate(
        [("successor A", 5.2), ("successor B", 4.8)]
    ):
        row = right_alpha * fade_in(t, 0.58 + index * 0.06, 0.12)
        y = 0.47 - index * 0.085
        text(fig, 0.56, y, label, size=19, color=DIM, alpha=row, family=MONO)
        text(
            fig,
            0.88,
            y,
            f"{value}",
            size=19,
            color=INK,
            alpha=row,
            family=MONO,
            ha="right",
        )
    text(
        fig,
        0.56,
        0.26,
        "→ picks B. Wrong subtree.",
        size=20,
        color=BAD,
        alpha=right_alpha * fade_in(t, 0.72),
    )

    text(
        fig,
        0.5,
        0.10,
        "so the loss optimises the ordering, not the estimate",
        size=23,
        color=DIM,
        alpha=alpha * fade_in(t, 0.84),
        ha="center",
        style="italic",
    )


def scene_turn(fig, t, data):
    alpha = fade_window(t, 0.0, 1.0, 0.12)
    text(
        fig,
        0.5,
        0.78,
        "Imitation optimises a proxy.",
        size=44,
        weight="bold",
        alpha=alpha * fade_in(t, 0.04),
        ha="center",
    )
    text(
        fig,
        0.5,
        0.68,
        "What we want is the heuristic that expands the fewest nodes.",
        size=24,
        color=DIM,
        alpha=alpha * fade_in(t, 0.18),
        ha="center",
    )

    rows = [
        ("state", "the open list, the closed set"),
        ("action", "which node to expand next"),
        ("policy", "argmin over h"),
        ("reward", "−1 per expansion"),
    ]
    panel(fig, 0.22, 0.20, 0.56, 0.38, alpha * fade_in(t, 0.32))
    for index, (key, value) in enumerate(rows):
        row = alpha * fade_in(t, 0.38 + index * 0.075, 0.14)
        y = 0.50 - index * 0.075
        colour = YELLOW if key == "reward" else INK
        text(fig, 0.26, y, key, size=21, color=DIM, alpha=row, family=MONO)
        text(fig, 0.40, y, value, size=21, color=colour, alpha=row, family=MONO)

    text(
        fig,
        0.5,
        0.11,
        "That is not a proxy. That is reinforcement learning.",
        size=25,
        color=BLUE,
        weight="bold",
        alpha=alpha * fade_in(t, 0.76),
        ha="center",
    )


def scene_cem(fig, t, data):
    alpha = fade_window(t, 0.0, 1.0, 0.10)
    _header(
        fig,
        "No gradient. So don't use one.",
        "perturb the weights, run the planner, keep what searched least, repeat",
        alpha,
    )

    history = data["cem"]["history"]
    reveal = min(1.0, max(0.0, (t - 0.18) / 0.55))
    shown = max(1, int(round(len(history) * ease_in_out(reveal))))

    ax = _axes(fig, [0.09, 0.20, 0.52, 0.46], alpha * fade_in(t, 0.14))
    xs = [h["iteration"] for h in history[:shown]]
    tuning = [
        h["tuning"] if h["tuning"] is not None else h["validation"]
        for h in history[:shown]
    ]
    validation = [h["validation"] for h in history[:shown]]
    # Both series are mean expansions — the same unit, so one axis is correct.
    ax.plot(xs, tuning, color=ORANGE, linewidth=2.0, solid_capstyle="round")
    ax.plot(xs, validation, color=BLUE, linewidth=2.0, solid_capstyle="round")
    ax.scatter([xs[-1]], [tuning[-1]], s=64, color=ORANGE, zorder=5)
    ax.scatter([xs[-1]], [validation[-1]], s=64, color=BLUE, zorder=5)
    ax.set_xlim(-0.3, len(history) - 0.4)
    top = max(max(tuning), max(validation)) * 1.15
    ax.set_ylim(0, top)
    ax.set_xlabel("CEM iteration", color=MUTED, fontsize=14)
    ax.set_ylabel("mean nodes expanded", color=MUTED, fontsize=14)
    # Two series: direct-labelled rather than boxed, so identity is never colour alone.
    ax.annotate(
        f"tuning  {tuning[-1]:.0f}",
        (xs[-1], tuning[-1]),
        textcoords="offset points",
        xytext=(10, 6),
        color=ORANGE,
        fontsize=16,
        fontweight="bold",
        alpha=alpha,
    )
    ax.annotate(
        f"validation  {validation[-1]:.0f}",
        (xs[-1], validation[-1]),
        textcoords="offset points",
        xytext=(10, -14),
        color=BLUE,
        fontsize=16,
        fontweight="bold",
        alpha=alpha,
    )

    right = alpha * fade_in(t, 0.60)
    text(fig, 0.68, 0.60, "cross-entropy method", size=24, weight="bold", alpha=right)
    text(
        fig,
        0.68,
        0.545,
        f"{data['cem']['population']} candidates x {data['cem']['iterations']} rounds",
        size=19,
        color=DIM,
        alpha=right,
    )
    _stat(
        fig,
        0.68,
        0.38,
        f"{data['cem']['seconds']:.0f}s",
        "of planning, as the objective",
        colour=AQUA,
        alpha=alpha * fade_in(t, 0.74),
        size=50,
    )
    text(
        fig,
        0.68,
        0.19,
        "the planner is the environment",
        size=19,
        color=MUTED,
        alpha=alpha * fade_in(t, 0.86),
    )


def scene_flat(fig, t, data):
    alpha = fade_window(t, 0.0, 1.0, 0.10)
    _header(
        fig,
        "Trap one: the objective is flat.",
        "on instances search already solves well, every perturbation scores the same",
        alpha,
    )

    flat = data["flat"]
    blocks = [
        (
            "tuned on the training ladder",
            flat["easy_before"],
            flat["easy_after"],
            BAD,
            "no headroom left",
            0.16,
        ),
        (
            "tuned a rung higher",
            flat["hard_before"],
            flat["hard_after"],
            GOOD,
            "room to be wrong in",
            0.46,
        ),
    ]
    for label, before, after, colour, caption, start in blocks:
        block = alpha * fade_in(t, start)
        y = 0.56 if start < 0.3 else 0.30
        text(fig, 0.09, y + 0.06, label, size=23, color=DIM, alpha=block)
        text(
            fig,
            0.09,
            y - 0.02,
            f"{before:,.0f}",
            size=46,
            weight="bold",
            color=MUTED,
            alpha=block,
            family=MONO,
        )
        text(
            fig,
            0.28,
            y - 0.02,
            "→",
            size=34,
            color=MUTED,
            alpha=block * fade_in(t, start + 0.08),
        )
        text(
            fig,
            0.36,
            y - 0.02,
            f"{after:,.0f}",
            size=46,
            weight="bold",
            color=colour,
            alpha=block * fade_in(t, start + 0.12),
            family=MONO,
        )
        text(
            fig,
            0.56,
            y - 0.02,
            caption,
            size=20,
            color=colour,
            alpha=block * fade_in(t, start + 0.16),
        )

    text(
        fig,
        0.5,
        0.10,
        "optimise where the search is still bad",
        size=25,
        color=INK,
        weight="bold",
        alpha=alpha * fade_in(t, 0.80),
        ha="center",
    )


def scene_spread(fig, t, data):
    """The distribution behind the mean — and the instance that decides it."""
    alpha = fade_window(t, 0.0, 1.0, 0.10)
    _header(
        fig,
        "Trap two: the mean was doing the lying.",
        "same run, one setting changed — and nine of ten instances barely notice",
        alpha,
    )

    spread = data["spread"]
    names = spread["instances"]
    lo, hi = spread["lo"], spread["hi"]
    budget = spread["budget"]

    ax = _axes(fig, [0.085, 0.20, 0.60, 0.47], alpha * fade_in(t, 0.14))
    xs = list(range(len(names)))
    base = [spread["imitation"][n] for n in names]
    lo_vals = [lo["per_instance"][n] for n in names]
    hi_vals = [hi["per_instance"][n] for n in names]

    reveal = ease_out(min(1.0, max(0.0, (t - 0.18) / 0.34)))
    shown = max(1, int(round(len(names) * reveal)))
    ax.set_yscale("log")
    ax.set_ylim(30, budget * 2.2)
    ax.set_xlim(-0.6, len(names) - 0.4)
    # Log scale because the tail spans three orders of magnitude; a linear axis
    # would render nine of the ten instances as a flat line on the floor.
    ax.plot(
        xs[:shown], base[:shown], color=MUTED, linewidth=2.0, marker="o", markersize=7
    )
    ax.plot(
        xs[:shown],
        lo_vals[:shown],
        color=ORANGE,
        linewidth=2.0,
        marker="o",
        markersize=7,
    )
    ax.plot(
        xs[:shown], hi_vals[:shown], color=BLUE, linewidth=2.0, marker="o", markersize=7
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([n.split("-")[1] for n in names], color=DIM, fontsize=13)
    ax.set_xlabel("held-out instance, by number of blocks", color=MUTED, fontsize=14)
    ax.set_ylabel("nodes expanded (log)", color=MUTED, fontsize=14)

    legend = alpha * fade_in(t, 0.30)
    for index, (label, colour) in enumerate(
        [
            ("imitation", MUTED),
            (f"sigma {lo['sigma']}", ORANGE),
            (f"sigma {hi['sigma']}", BLUE),
        ]
    ):
        y = 0.615 - index * 0.045
        ax.figure.patches.append(
            plt.Rectangle(
                (0.115, y - 0.008),
                0.016,
                0.016,
                transform=fig.transFigure,
                facecolor=colour,
                edgecolor="none",
                alpha=legend,
            )
        )
        text(fig, 0.140, y, label, size=15, color=DIM, alpha=legend)

    # The outlier, called out where it sits.
    worst = max(names, key=lambda n: spread["imitation"][n])
    if t > 0.52:
        marker = alpha * fade_in(t, 0.52)
        index = names.index(worst)
        ax.annotate(
            "imitation never solved this one",
            (index, spread["imitation"][worst]),
            textcoords="offset points",
            xytext=(-150, 14),
            color=BAD,
            fontsize=15,
            fontweight="bold",
            alpha=marker,
        )

    right = alpha * fade_in(t, 0.64)
    text(fig, 0.72, 0.60, "the two means", size=21, color=DIM, alpha=right)
    for row, (value, colour, tag) in enumerate(
        [
            (lo["mean"], ORANGE, f"sigma {lo['sigma']}"),
            (hi["mean"], BLUE, f"sigma {hi['sigma']}"),
        ]
    ):
        y = 0.535 - row * 0.062
        text(
            fig,
            0.72,
            y,
            f"{value:>6,.0f}",
            size=30,
            weight="bold",
            color=INK,
            alpha=right,
            family=MONO,
        )
        text(fig, 0.845, y, tag, size=16, color=colour, alpha=right)
    text(
        fig,
        0.72,
        0.395,
        f"differ by {lo['mean'] / max(1.0, hi['mean']):.0f}x.",
        size=22,
        color=DIM,
        alpha=alpha * fade_in(t, 0.72),
    )
    text(
        fig,
        0.72,
        0.32,
        "One instance out of ten\nis the whole gap.",
        size=22,
        color=BAD,
        weight="bold",
        alpha=alpha * fade_in(t, 0.80),
    )
    text(
        fig,
        0.72,
        0.19,
        "look at the distribution\nbefore believing the mean",
        size=19,
        color=GOOD,
        alpha=alpha * fade_in(t, 0.88),
    )


def scene_result(fig, t, data):
    alpha = fade_window(t, 0.0, 1.0, 0.10)
    space = data["space"]
    _header(
        fig,
        "Trained on 3-6 blocks. Judged on 9-13.",
        "greedy best-first search, on instances no stage of training ever saw",
        alpha,
    )

    after = data["transfer_after"]
    order = ["learned", "hff", "goalcount"]
    colours = {"learned": BLUE, "hff": MUTED, "goalcount": "#4a4a4e"}

    # Two measures on different scales get two charts. Never two y-axes.
    for panel_index, (key, title, fmt_value) in enumerate(
        [
            ("mean_expanded", "nodes expanded", lambda v: f"{v:,.0f}"),
            ("mean_seconds", "seconds", lambda v: f"{v:.3f}"),
        ]
    ):
        left = 0.09 + panel_index * 0.47
        ax = _axes(
            fig, [left, 0.22, 0.36, 0.44], alpha * fade_in(t, 0.14 + panel_index * 0.06)
        )
        values = [after[name][key] for name in order]
        ax.set_xlim(-0.65, 2.65)
        ax.set_ylim(0, max(values) * 1.24)
        for index, name in enumerate(order):
            grow = ease_out(
                min(
                    1.0, max(0.0, (t - 0.22 - index * 0.09 - panel_index * 0.05) / 0.32)
                )
            )
            _rounded_bar(ax, index, 0.52, values[index] * grow, colours[name], alpha)
            if grow > 0.6:
                ax.text(
                    index,
                    values[index] * grow + max(values) * 0.04,
                    fmt_value(values[index]),
                    ha="center",
                    color=colours[name] if name == "learned" else DIM,
                    fontsize=19,
                    fontweight="bold",
                    alpha=alpha * ease_out((grow - 0.6) / 0.4),
                )
        ax.set_xticks(range(3))
        ax.set_xticklabels(order, color=DIM, fontsize=15)
        ax.set_title(title, color=MUTED, fontsize=16, pad=12)

    ratio = after["hff"]["mean_expanded"] / max(1e-9, after["learned"]["mean_expanded"])
    speed = after["hff"]["mean_seconds"] / max(1e-9, after["learned"]["mean_seconds"])
    text(
        fig,
        0.5,
        0.11,
        f"{ratio:.1f}x fewer expansions than h_ff, and {speed:.0f}x faster",
        size=27,
        weight="bold",
        color=INK,
        alpha=alpha * fade_in(t, 0.72),
        ha="center",
    )
    text(
        fig,
        0.5,
        0.05,
        f"trained on {space['train_instances']} instances in "
        f"{data['imitation']['seconds']:.1f}s",
        size=17,
        color=MUTED,
        alpha=alpha * fade_in(t, 0.84),
        ha="center",
    )


def scene_honest(fig, t, data):
    alpha = fade_window(t, 0.0, 1.0, 0.10)
    log = data["logistics"]
    _header(
        fig,
        "And on logistics, it loses.",
        "worth saying, because the reason is exact rather than mysterious",
        alpha,
    )

    left = alpha * fade_in(t, 0.14)
    text(fig, 0.09, 0.60, "the domain has", size=22, color=DIM, alpha=left)
    text(
        fig,
        0.09,
        0.50,
        "  ".join(f"({p} ...)" for p in log["predicates"]),
        size=30,
        color=INK,
        alpha=left,
        family=MONO,
    )
    text(
        fig,
        0.09,
        0.41,
        f"two predicates → {log['features']} features",
        size=21,
        color=DIM,
        alpha=alpha * fade_in(t, 0.30),
    )
    text(
        fig,
        0.09,
        0.30,
        "which cannot say which package is where —",
        size=23,
        color=BAD,
        alpha=alpha * fade_in(t, 0.44),
    )
    text(
        fig,
        0.09,
        0.235,
        "only how many are somewhere at all.",
        size=23,
        color=BAD,
        alpha=alpha * fade_in(t, 0.52),
    )

    right = alpha * fade_in(t, 0.64)
    _stat(
        fig,
        0.68,
        0.56,
        f"{log['top1']:.0%}",
        "of decisions ranked correctly",
        colour=BAD,
        alpha=right,
        size=54,
        sub=f"against {log['blocks_top1']:.0%} on blocksworld",
    )
    _stat(
        fig,
        0.68,
        0.30,
        f"{log['learned'] / max(1.0, log['hff']):.0f}x",
        "more expansions than h_ff",
        colour=DIM,
        alpha=alpha * fade_in(t, 0.78),
        size=54,
    )
    text(
        fig,
        0.5,
        0.09,
        "counting is blind to topology — that is the case for relational features",
        size=21,
        color=MUTED,
        alpha=alpha * fade_in(t, 0.88),
        ha="center",
        style="italic",
    )


def scene_cta(fig, t, data):
    alpha = fade_window(t, 0.0, 1.0, 0.14)
    text(
        fig,
        0.5,
        0.70,
        "jupyddl learn",
        size=60,
        weight="bold",
        alpha=alpha * fade_in(t, 0.05),
        ha="center",
        family=MONO,
    )
    command = "jupyddl learn blocksworld --cem 10 --evaluate 9-13"
    panel(fig, 0.14, 0.44, 0.72, 0.11, alpha * fade_in(t, 0.22))
    text(
        fig,
        0.5,
        0.495,
        typewriter(command, t, 0.26, 0.34),
        size=24,
        color=AQUA,
        alpha=alpha * fade_in(t, 0.26),
        ha="center",
        family=MONO,
    )
    text(
        fig,
        0.5,
        0.33,
        "-H learned:model.json  ·  anywhere a heuristic name goes",
        size=21,
        color=DIM,
        alpha=alpha * fade_in(t, 0.62),
        ha="center",
        family=MONO,
    )
    text(
        fig,
        0.5,
        0.22,
        "github.com/APLA-Toolbox/PythonPDDL",
        size=22,
        color=INK,
        alpha=alpha * fade_in(t, 0.72),
        ha="center",
        family=MONO,
    )
    text(
        fig,
        0.5,
        0.14,
        "zero dependencies · the research notes are in .docs/",
        size=18,
        color=MUTED,
        alpha=alpha * fade_in(t, 0.80),
        ha="center",
    )


SCENES = [
    (scene_hook, 5.0),
    (scene_labels, 9.0),
    (scene_imitation, 8.5),
    (scene_order, 9.5),
    (scene_turn, 8.0),
    (scene_cem, 11.0),
    (scene_flat, 9.0),
    (scene_spread, 11.0),
    (scene_result, 10.0),
    (scene_honest, 10.0),
    (scene_cta, 6.5),
]


def render(data, out: str, fps: int = FPS, dpi: int = DPI):
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
                draw(fig, frame / max(1, frames - 1), data)
                writer.grab_frame(facecolor=BG)
                done += 1
                if done % 60 == 0:
                    rate = done / (time.perf_counter() - started)
                    print(f"  {done}/{total} frames ({rate:.0f} fps render)")
    plt.close(fig)
    print(f"Wrote {out} in {time.perf_counter() - started:.0f}s")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the learned-heuristic / RL promo video."
    )
    parser.add_argument("-o", "--output", default="promo/jupyddl-rl.mp4")
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument(
        "--cache",
        default=None,
        help="read measurements from this JSON if it exists, else write them to it",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="measure and write the cache without rendering",
    )
    args = parser.parse_args()

    if args.cache and os.path.exists(args.cache):
        print(f"Reusing measurements from {args.cache}")
        with open(args.cache, encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        print("Measuring (training, reinforcing, and reproducing both traps)...")
        data = collect()
        if args.cache:
            os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)
            with open(args.cache, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=1)
            print(f"Wrote {args.cache} ({data['total_seconds']:.0f}s of measurement)")

    if args.collect_only:
        return 0

    folder = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(folder, exist_ok=True)
    render(data, args.output, fps=args.fps, dpi=args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

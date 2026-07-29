"""Static charts for search traces and benchmark runs.

Every function takes recorded data (a :class:`~jupyddl.trace.SearchTrace`, a list
of traces, or benchmark rows), returns the matplotlib ``Figure``, and optionally
writes it to ``path``. Pass ``dark=True`` for the dark-surface variant of the
same palette — the steps are chosen for that surface, not flipped.

Requires the ``viz`` extra::

    pip install -e ".[viz]"
"""

from __future__ import annotations

import math
from typing import Optional

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "jupyddl.viz needs matplotlib. Install the extra: pip install 'jupyddl[viz]'"
    ) from exc

from .theme import palette, rc_params, sequential, sequential_cmap, series_color

__all__ = [
    "plot_search_progress",
    "plot_planner_comparison",
    "plot_search_tree",
    "plot_benchmark_dashboard",
    "plot_plan_timeline",
    "rounded_bars",
]

# Cubic-Bezier circle constant: a quarter turn needs control points at 0.5523r.
_KAPPA = 0.5523


# --------------------------------------------------------------------------
# figure scaffolding
# --------------------------------------------------------------------------
def _header(fig, title: str, subtitle: str, pal, reserve: float = 0.90) -> None:
    """Put a two-level header above the axes and reserve room for it.

    ``constrained`` layout only budgets for ``suptitle``; reserving an explicit
    rect keeps hand-placed header text from landing on the panel titles.
    """
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0, 0, 1, reserve))
    fig.text(
        0.008,
        0.988,
        title,
        fontsize=13,
        fontweight="bold",
        color=pal["text"],
        ha="left",
        va="top",
    )
    if subtitle:
        fig.text(
            0.008,
            0.988 - (1 - reserve) * 0.52,
            subtitle,
            fontsize=9,
            color=pal["text_secondary"],
            ha="left",
            va="top",
        )


def _finish(fig, path: Optional[str]):
    """Save (if asked) and hand the figure back.

    Writing a file also releases the figure from pyplot's registry: rendering a
    gallery of dozens of charts would otherwise hold every one of them in memory
    until the process exits. The returned figure is still usable — call
    ``fig.savefig(...)`` again if you want another copy — and callers who want to
    keep it managed by pyplot should pass ``path=None`` and save it themselves.
    """
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def _credit(fig, pal, text="jupyddl"):
    fig.text(
        0.995, 0.005, text, ha="right", va="bottom", fontsize=7, color=pal["muted"]
    )


# --------------------------------------------------------------------------
# rounded bars
# --------------------------------------------------------------------------
def _pixel_radius_in_data(ax, x: float, y: float, radius_px: float):
    """Convert a screen-space radius to data units *at this point*.

    Doing the conversion through ``transData`` (rather than assuming a linear
    scale) keeps the corner a true circle on log axes too.
    """
    to_display = ax.transData.transform
    to_data = ax.transData.inverted().transform
    px, py = to_display((x, y))
    rx = abs(to_data((px + radius_px, py))[0] - x)
    ry = abs(to_data((px, py + radius_px))[1] - y)
    return rx, ry


def _rounded_rect_path(x0, y0, width, height, rx, ry, side: str) -> Path:
    """Rectangle with two rounded corners on ``side`` and a square opposite end."""
    rx = max(0.0, min(rx, abs(width) / 2.0))
    ry = max(0.0, min(ry, abs(height) / 2.0))
    x1, y1 = x0 + width, y0 + height
    k = _KAPPA
    verts, codes = [], []

    def start(point):
        verts.append(point)
        codes.append(Path.MOVETO)

    def line(point):
        verts.append(point)
        codes.append(Path.LINETO)

    def curve(c1, c2, end):
        verts.extend([c1, c2, end])
        codes.extend([Path.CURVE4] * 3)

    if side == "top":  # vertical bar growing up
        start((x0, y0))
        line((x0, y1 - ry))
        curve((x0, y1 - ry + ry * k), (x0 + rx - rx * k, y1), (x0 + rx, y1))
        line((x1 - rx, y1))
        curve((x1 - rx + rx * k, y1), (x1, y1 - ry + ry * k), (x1, y1 - ry))
        line((x1, y0))
    elif side == "bottom":  # vertical bar growing down
        start((x0, y0))
        line((x0, y1 + ry))
        curve((x0, y1 + ry - ry * k), (x0 + rx - rx * k, y1), (x0 + rx, y1))
        line((x1 - rx, y1))
        curve((x1 - rx + rx * k, y1), (x1, y1 + ry - ry * k), (x1, y1 + ry))
        line((x1, y0))
    elif side == "right":  # horizontal bar growing right
        start((x0, y0))
        line((x1 - rx, y0))
        curve((x1 - rx + rx * k, y0), (x1, y0 + ry - ry * k), (x1, y0 + ry))
        line((x1, y1 - ry))
        curve((x1, y1 - ry + ry * k), (x1 - rx + rx * k, y1), (x1 - rx, y1))
        line((x0, y1))
    else:  # "left"
        start((x0, y0))
        line((x1 + rx, y0))
        curve((x1 + rx - rx * k, y0), (x1, y0 + ry - ry * k), (x1, y0 + ry))
        line((x1, y1 - ry))
        curve((x1, y1 - ry + ry * k), (x1 + rx - rx * k, y1), (x1 + rx, y1))
        line((x0, y1))

    verts.append(verts[0])
    codes.append(Path.CLOSEPOLY)
    return Path(verts, codes)


def rounded_bars(
    ax,
    positions,
    values,
    colors,
    width: float = 0.62,
    radius_px: float = 4.0,
    baseline: Optional[float] = None,
    surface: str = "#fcfcfb",
    horizontal: bool = False,
):
    """Draw bars with a rounded data-end and a square baseline.

    The corner is a true ``radius_px`` circle in screen space at the bar's own
    location, so it stays circular on log axes and under any figure size. A
    hairline in the surface colour supplies the 2px gap between adjacent bars.
    """
    fig = ax.figure
    fig.canvas.draw()  # resolve the layout before reading pixel scales
    if baseline is None:
        baseline = (ax.get_xlim() if horizontal else ax.get_ylim())[0]

    patches = []
    for pos, value, color in zip(positions, values, colors):
        if value is None or value != value:
            continue
        span = value - baseline
        if span == 0:
            continue
        if horizontal:
            rx, ry = _pixel_radius_in_data(ax, value, pos, radius_px)
            side = "right" if span > 0 else "left"
            path = _rounded_rect_path(
                baseline, pos - width / 2.0, span, width, rx, ry, side
            )
        else:
            rx, ry = _pixel_radius_in_data(ax, pos, value, radius_px)
            side = "top" if span > 0 else "bottom"
            path = _rounded_rect_path(
                pos - width / 2.0, baseline, width, span, rx, ry, side
            )
        patch = PathPatch(
            path,
            facecolor=color,
            edgecolor=surface,
            linewidth=1.6,
            joinstyle="round",
            zorder=3,
        )
        ax.add_patch(patch)
        patches.append(patch)
    return patches


# --------------------------------------------------------------------------
# labels & formatting
# --------------------------------------------------------------------------
def _fmt(value) -> str:
    """Compact human formatting for counts."""
    if value is None:
        return "–"
    if isinstance(value, float) and not float(value).is_integer():
        return f"{value:.2f}" if value < 10 else f"{value:,.0f}"
    value = int(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 10_000:
        return f"{value / 1000:.0f}k"
    return f"{value:,}"


def _label_bars(ax, positions, values, pal, horizontal=False, fmt=_fmt):
    """Direct value labels — magnitude never relies on colour alone."""
    for pos, value in zip(positions, values):
        if value is None or value != value:
            continue
        xy = (value, pos) if horizontal else (pos, value)
        ax.annotate(
            fmt(value),
            xy,
            xytext=(5, 0) if horizontal else (0, 4),
            textcoords="offset points",
            va="center" if horizontal else "bottom",
            ha="left" if horizontal else "center",
            fontsize=8,
            color=pal["text_secondary"],
            clip_on=False,
            zorder=6,
        )


def _headroom(ax, values, factor=1.2):
    finite = [v for v in values if v is not None and v == v]
    top = max(finite or [1])
    ax.set_ylim(0, top * factor if top > 0 else 1)


def _stagger_labels(ax, entries, pal):
    """Direct-label line ends, spread vertically so they never overprint."""
    count = len(entries)
    for i, (x, y, text, color) in enumerate(entries):
        offset = (i - (count - 1) / 2.0) * 12
        ax.annotate(
            text,
            (x, y),
            xytext=(-6, offset),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=color,
            clip_on=False,
            zorder=6,
        )


# --------------------------------------------------------------------------
# 1. one search, in detail
# --------------------------------------------------------------------------
def plot_search_progress(
    trace, path: Optional[str] = None, dark: bool = False, title: Optional[str] = None
):
    """Four views of a single search: cost estimates, frontier, depth, throughput."""
    pal = palette(dark)
    with plt.rc_context(rc_params(dark)):
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), layout="constrained")
        expansions = trace.expansions
        steps = list(range(1, len(expansions) + 1))
        summary = trace.summary()
        verdict = (
            f"solved · cost {summary['cost']} · {summary['plan_length']} actions"
            if summary["solved"]
            else "no plan found"
        )
        _header(
            fig,
            title or f"{trace.label} on {trace.task_name or 'task'}",
            f"{verdict}   ·   {_fmt(summary['expanded'])} expanded   ·   "
            f"{_fmt(summary['generated'])} generated   ·   "
            f"{summary['runtime'] * 1000:.0f} ms",
            pal,
            reserve=0.90,
        )

        if not expansions:
            for ax in axes.flat:
                ax.text(
                    0.5,
                    0.5,
                    "no expansions recorded",
                    ha="center",
                    va="center",
                    color=pal["muted"],
                    transform=ax.transAxes,
                )
            return _finish(fig, path)

        # --- A: the f = g + h decomposition over expansion order -------------
        ax = axes[0][0]
        gs = [e.g for e in expansions]
        hs = [e.h for e in expansions]
        fs = [e.f for e in expansions]
        ax.plot(steps, fs, color=series_color(0, dark), label="f = g + h", zorder=4)
        ax.plot(steps, gs, color=series_color(1, dark), label="g (path cost)", zorder=3)
        ax.plot(steps, hs, color=series_color(2, dark), label="h (estimate)", zorder=3)
        _headroom(ax, fs + gs + hs, 1.42)  # room for the legend above the lines
        # No end-labels here: g rises to meet f while h falls to zero, so all
        # three converge on the right edge. The legend carries identity.
        ax.set_title("Cost estimates as the search advances")
        ax.set_xlabel("nodes expanded")
        ax.set_ylabel("cost")
        ax.legend(loc="upper left", ncol=3)

        # --- B: frontier size = the memory profile ---------------------------
        ax = axes[0][1]
        opens = [e.open_size for e in expansions]
        ax.plot(steps, opens, color=series_color(0, dark), zorder=3)
        ax.fill_between(steps, opens, color=series_color(0, dark), alpha=0.12, zorder=2)
        _headroom(ax, opens, 1.22)
        peak = max(opens) if opens else 0
        if peak:
            idx = opens.index(peak)
            ax.annotate(
                f"peak {_fmt(peak)}",
                (steps[idx], peak),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=pal["text_secondary"],
            )
        ax.set_title("Frontier size (open list)")
        ax.set_xlabel("nodes expanded")
        ax.set_ylabel("nodes on the frontier")

        # --- C: where in the space the search is looking ---------------------
        ax = axes[1][0]
        depths = [e.depth for e in expansions]
        ax.plot(steps, depths, color=series_color(3, dark), zorder=3)
        _headroom(ax, depths, 1.25)
        bounds = trace.bounds()
        for expanded_at, _threshold in bounds:
            ax.axvline(
                max(1, expanded_at),
                color=pal["axis"],
                linewidth=1,
                linestyle=(0, (3, 3)),
                zorder=1,
            )
        if bounds:
            ax.annotate(
                f"{len(bounds)} bound restarts",
                (0.98, 0.94),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=8,
                color=pal["muted"],
            )
        ax.set_title("Depth of the expanded node")
        ax.set_xlabel("nodes expanded")
        ax.set_ylabel("depth")

        # --- D: throughput ---------------------------------------------------
        ax = axes[1][1]
        elapsed = [e.elapsed * 1000 for e in expansions]
        generated = [e.generated for e in expansions]
        ax.plot(elapsed, steps, color=series_color(0, dark), label="expanded", zorder=4)
        ax.plot(
            elapsed, generated, color=series_color(1, dark), label="generated", zorder=3
        )
        _headroom(ax, steps + generated, 1.32)
        _stagger_labels(
            ax,
            [
                (elapsed[-1], generated[-1], "generated", series_color(1, dark)),
                (elapsed[-1], steps[-1], "expanded", series_color(0, dark)),
            ],
            pal,
        )
        ax.set_title("Nodes touched over time")
        ax.set_xlabel("milliseconds")
        ax.set_ylabel("cumulative nodes")
        ax.legend(loc="upper left")
        _credit(fig, pal)
    return _finish(fig, path)


# --------------------------------------------------------------------------
# 2. several searches, compared
# --------------------------------------------------------------------------
def plot_planner_comparison(
    traces, path: Optional[str] = None, dark: bool = False, title: Optional[str] = None
):
    """Compare configurations on the same task: effort, time, quality, guidance."""
    pal = palette(dark)
    traces = list(traces)
    labels = [t.label for t in traces]
    colors = [series_color(i, dark) for i in range(len(traces))]
    positions = list(range(len(traces)))

    with plt.rc_context(rc_params(dark)):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.0), layout="constrained")
        task = traces[0].task_name if traces else ""
        _header(
            fig,
            title or f"Planner comparison — {task}",
            "same task, same machine · lower is better on every panel but the last",
            pal,
            reserve=0.91,
        )

        # --- A: search effort (log scale: the spread is orders of magnitude) --
        ax = axes[0][0]
        expanded = [t.stats.get("expanded", 0) or 0 for t in traces]
        ax.set_yscale("log")
        ax.set_xlim(-0.6, len(traces) - 0.4)
        ax.set_ylim(0.7, max(expanded + [10]) * 4)
        rounded_bars(
            ax,
            positions,
            [max(v, 1) for v in expanded],
            colors,
            baseline=0.7,
            surface=pal["surface"],
        )
        _label_bars(ax, positions, expanded, pal)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_title("Nodes expanded  (log scale)")
        ax.set_ylabel("nodes")

        # --- B: runtime ------------------------------------------------------
        ax = axes[0][1]
        runtime = [(t.stats.get("runtime", 0.0) or 0.0) * 1000 for t in traces]
        ax.set_xlim(-0.6, len(traces) - 0.4)
        _headroom(ax, runtime)
        rounded_bars(ax, positions, runtime, colors, surface=pal["surface"])
        _label_bars(
            ax,
            positions,
            runtime,
            pal,
            fmt=lambda v: f"{v:.0f} ms" if v >= 1 else f"{v:.1f} ms",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_title("Wall-clock runtime")
        ax.set_ylabel("milliseconds")

        # --- C: plan quality -------------------------------------------------
        ax = axes[1][0]
        costs = [t.cost if t.solved and t.cost is not None else 0 for t in traces]
        best = min([c for c in costs if c > 0] or [0])
        ax.set_xlim(-0.6, len(traces) - 0.4)
        _headroom(ax, costs, 1.28)
        rounded_bars(ax, positions, costs, colors, surface=pal["surface"])
        _label_bars(ax, positions, costs, pal)
        if best:
            ax.axhline(
                best, color=pal["axis"], linewidth=1, linestyle=(0, (3, 3)), zorder=2
            )
            ax.annotate(
                f"best = {best}",
                (len(traces) - 0.45, best),
                xytext=(0, 5),
                textcoords="offset points",
                ha="right",
                fontsize=8,
                color=pal["muted"],
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_title("Plan cost  (lower is better)")
        ax.set_ylabel("cost")

        # --- D: how the heuristic actually guided ----------------------------
        ax = axes[1][1]
        ends = []
        for i, trace in enumerate(traces):
            hs = [e.h for e in trace.expansions]
            if not hs or max(hs) == 0:
                continue
            xs = [j / max(1, len(hs) - 1) * 100 for j in range(len(hs))]
            ax.plot(xs, hs, color=colors[i], label=trace.label, zorder=3)
            ends.append((xs[-1], hs[-1], trace.label, colors[i]))
        if ends:
            # Every run finishes near h = 0, so end-labels would pile up on one
            # point; identity comes from the legend instead.
            ax.legend(loc="upper right")
            ax.set_title("Heuristic estimate along the search")
            ax.set_xlabel("search progress (%)")
            ax.set_ylabel("h of the expanded node")
        else:
            ax.text(
                0.5,
                0.5,
                "no heuristic values recorded",
                ha="center",
                va="center",
                color=pal["muted"],
                transform=ax.transAxes,
            )
            ax.set_title("Heuristic estimate along the search")
        _credit(fig, pal)
    return _finish(fig, path)


# --------------------------------------------------------------------------
# 3. the shape of the search itself
# --------------------------------------------------------------------------
def plot_search_tree(
    trace,
    path: Optional[str] = None,
    dark: bool = False,
    max_nodes: int = 4000,
    title: Optional[str] = None,
):
    """A radial wavefront of the expanded nodes: radius = depth, colour = h.

    Each ring is one depth level and each spoke one expanded node, so a
    well-guided search reads as a narrow spike and a blind one as a full disc.
    """
    pal = palette(dark)
    expansions = trace.expansions[:max_nodes]

    with plt.rc_context(rc_params(dark)):
        fig, ax = plt.subplots(
            figsize=(7.6, 7.4), subplot_kw={"projection": "polar"}, layout="constrained"
        )
        ax.set_facecolor(pal["surface"])
        ax.grid(color=pal["grid"], linewidth=0.7)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.spines["polar"].set_color(pal["axis"])
        _header(
            fig,
            title or f"Search wavefront — {trace.label}",
            f"{trace.task_name} · {len(expansions):,} expanded nodes · "
            f"ring = depth · colour = heuristic estimate",
            pal,
            reserve=0.92,
        )

        if not expansions:
            ax.text(0, 0, "no expansions", ha="center", va="center", color=pal["muted"])
            return _finish(fig, path)

        # Lay nodes out ring by ring: one ring per depth, evenly spread.
        by_depth: dict = {}
        for event in expansions:
            by_depth.setdefault(event.depth, []).append(event)
        angle_of: dict = {}
        radius_of: dict = {}
        for depth, events in by_depth.items():
            count = len(events)
            for i, event in enumerate(events):
                angle_of[event.node] = 2 * math.pi * (i + 0.5) / count
                radius_of[event.node] = depth

        h_values = [e.h for e in expansions]
        h_max = max(h_values) if h_values else 0

        for event in expansions:  # edges first, so the nodes sit on top
            if event.parent < 0 or event.parent not in angle_of:
                continue
            ax.plot(
                [angle_of[event.parent], angle_of[event.node]],
                [radius_of[event.parent], radius_of[event.node]],
                color=pal["axis"],
                linewidth=0.5,
                alpha=0.55,
                zorder=2,
                solid_capstyle="round",
            )

        ax.scatter(
            [angle_of[e.node] for e in expansions],
            [radius_of[e.node] for e in expansions],
            s=26,
            c=[sequential(e.h / h_max if h_max else 0.5, dark) for e in expansions],
            zorder=4,
            edgecolors=pal["surface"],
            linewidths=0.6,
        )

        # Best-first planners recognise the goal when they pop it, so the goal
        # state is usually never *expanded* and has no ring position of its own.
        # Place it on its own depth ring — the angle carries no meaning in this
        # layout for any node, so an unused spoke is consistent, not misleading.
        goals = trace.of_kind("goal")
        for goal in goals:
            if goal.node in angle_of:
                angle, radius = angle_of[goal.node], radius_of[goal.node]
            else:
                angle, radius = 0.0, goal.depth
            ax.scatter(
                [angle],
                [radius],
                s=190,
                marker="*",
                color=pal["good"],
                zorder=6,
                edgecolors=pal["surface"],
                linewidths=0.8,
            )

        legend = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=7,
                markerfacecolor=sequential(0.15, dark),
                markeredgecolor=pal["surface"],
                label="low h (near the goal)",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=7,
                markerfacecolor=sequential(0.95, dark),
                markeredgecolor=pal["surface"],
                label="high h (far away)",
            ),
        ]
        if goals:
            legend.append(
                Line2D(
                    [],
                    [],
                    marker="*",
                    linestyle="none",
                    markersize=11,
                    markerfacecolor=pal["good"],
                    markeredgecolor=pal["surface"],
                    label="goal reached",
                )
            )
        ax.legend(handles=legend, loc="upper right", bbox_to_anchor=(1.14, 1.06))
        _credit(fig, pal)
    return _finish(fig, path)


# --------------------------------------------------------------------------
# 4. a whole benchmark
# --------------------------------------------------------------------------
def plot_benchmark_dashboard(
    rows, path: Optional[str] = None, dark: bool = False, title: Optional[str] = None
):
    """Coverage, effort, time and a per-instance heatmap for a benchmark run."""
    pal = palette(dark)
    configs: list = []
    instances: list = []
    for row in rows:
        key = f"{row.planner}/{row.heuristic}" if row.heuristic else row.planner
        if key not in configs:
            configs.append(key)
        if row.instance not in instances:
            instances.append(row.instance)

    lookup = {}
    for row in rows:
        key = f"{row.planner}/{row.heuristic}" if row.heuristic else row.planner
        lookup[(row.instance, key)] = row

    colors = [series_color(i, dark) for i in range(len(configs))]
    positions = list(range(len(configs)))

    with plt.rc_context(rc_params(dark)):
        fig = plt.figure(figsize=(13, 8.0), layout="constrained")
        _header(
            fig,
            title or "jupyddl benchmark",
            f"{len(instances)} instances × {len(configs)} configurations",
            pal,
            reserve=0.92,
        )
        grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.3])

        # --- coverage ---------------------------------------------------------
        ax = fig.add_subplot(grid[0, 0])
        coverage = []
        for cfg in configs:
            hits = 0
            for inst in instances:
                record = lookup.get((inst, cfg))
                if record is not None and record.valid:
                    hits += 1
            coverage.append(hits)
        ax.set_xlim(-0.6, len(configs) - 0.4)
        ax.set_ylim(0, (len(instances) or 1) * 1.25)
        rounded_bars(
            ax, positions, coverage, colors, baseline=0, surface=pal["surface"]
        )
        _label_bars(
            ax, positions, coverage, pal, fmt=lambda v: f"{int(v)}/{len(instances)}"
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(configs, rotation=22, ha="right")
        ax.set_title("Coverage  (validated plans)")
        ax.set_ylabel("instances solved")

        # --- effort ------------------------------------------------------------
        ax = fig.add_subplot(grid[0, 1])
        expanded = []
        for cfg in configs:
            total = 0
            for inst in instances:
                record = lookup.get((inst, cfg))
                total += record.expanded if record else 0
            expanded.append(total)
        ax.set_yscale("log")
        ax.set_xlim(-0.6, len(configs) - 0.4)
        ax.set_ylim(0.7, max(expanded + [10]) * 4)
        rounded_bars(
            ax,
            positions,
            [max(v, 1) for v in expanded],
            colors,
            baseline=0.7,
            surface=pal["surface"],
        )
        _label_bars(ax, positions, expanded, pal)
        ax.set_xticks(positions)
        ax.set_xticklabels(configs, rotation=22, ha="right")
        ax.set_title("Total nodes expanded  (log scale)")
        ax.set_ylabel("nodes")

        # --- runtime -----------------------------------------------------------
        ax = fig.add_subplot(grid[0, 2])
        runtime = []
        for cfg in configs:
            total = 0.0
            for inst in instances:
                record = lookup.get((inst, cfg))
                total += record.runtime if record else 0.0
            runtime.append(total)
        ax.set_xlim(-0.6, len(configs) - 0.4)
        _headroom(ax, runtime)
        rounded_bars(ax, positions, runtime, colors, baseline=0, surface=pal["surface"])
        _label_bars(ax, positions, runtime, pal, fmt=lambda v: f"{v:.2f}s")
        ax.set_xticks(positions)
        ax.set_xticklabels(configs, rotation=22, ha="right")
        ax.set_title("Total runtime")
        ax.set_ylabel("seconds")

        # --- per-instance heatmap ---------------------------------------------
        ax = fig.add_subplot(grid[1, :])
        matrix, annotations = [], []
        for cfg in configs:
            row_vals, row_text = [], []
            for inst in instances:
                record = lookup.get((inst, cfg))
                if record is None or not record.valid:
                    row_vals.append(float("nan"))
                    row_text.append("×" if record is not None else "")
                else:
                    row_vals.append(math.log10(max(record.expanded, 1)))
                    row_text.append(_fmt(record.expanded))
            matrix.append(row_vals)
            annotations.append(row_text)

        finite = [v for row in matrix for v in row if v == v]
        vmin, vmax = (min(finite), max(finite)) if finite else (0.0, 1.0)
        # Cells with no valid plan are NaN and render as the page colour.
        cmap = sequential_cmap(dark).with_extremes(bad=pal["page"])
        image = ax.imshow(
            matrix,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_xticks(range(len(instances)))
        ax.set_xticklabels(instances, rotation=18, ha="right")
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(configs)
        ax.grid(False)
        ax.set_title("Nodes expanded per instance  ·  × = no valid plan")

        for i, row_text in enumerate(annotations):
            for j, text in enumerate(row_text):
                value = matrix[i][j]
                if value != value:
                    color = pal["muted"]
                else:
                    ratio = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    color = "#ffffff" if ratio > 0.55 else "#0b0b0b"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

        bar = fig.colorbar(image, ax=ax, pad=0.012, fraction=0.022)
        bar.set_label("log₁₀ nodes expanded", color=pal["text_secondary"], fontsize=8.5)
        bar.ax.tick_params(colors=pal["muted"], labelsize=8)
        bar.outline.set_edgecolor(pal["axis"])
        _credit(fig, pal)
    return _finish(fig, path)


# --------------------------------------------------------------------------
# 5. the plan that came out
# --------------------------------------------------------------------------
def plot_plan_timeline(
    trace,
    path: Optional[str] = None,
    dark: bool = False,
    max_steps: int = 40,
    title: Optional[str] = None,
):
    """The plan as an ordered list of actions with the cost accumulating."""
    pal = palette(dark)
    plan = list(trace.plan)
    truncated = len(plan) > max_steps
    shown = plan[:max_steps]

    with plt.rc_context(rc_params(dark)):
        height = max(3.2, 0.3 * len(shown) + 1.8)
        fig, ax = plt.subplots(figsize=(8.8, height), layout="constrained")
        caption = f"{len(plan)} actions · cost {trace.cost}"
        if truncated:
            caption += f" · first {max_steps} shown"
        _header(
            fig,
            title or f"Plan — {trace.label} on {trace.task_name}",
            caption,
            pal,
            reserve=1 - 1.05 / height,
        )

        if not shown:
            ax.text(
                0.5,
                0.5,
                "no plan",
                ha="center",
                va="center",
                color=pal["muted"],
                transform=ax.transAxes,
            )
            ax.axis("off")
            return _finish(fig, path)

        positions = list(range(len(shown)))
        steps = [i + 1 for i in positions]
        ax.set_xlim(0, len(shown) + 0.6)
        ax.set_ylim(len(shown) - 0.4, -0.6)  # inverted: step 1 on top
        rounded_bars(
            ax,
            positions,
            steps,
            [
                sequential(0.35 + 0.55 * i / max(1, len(shown) - 1), dark)
                for i in positions
            ],
            width=0.6,
            baseline=0,
            surface=pal["surface"],
            horizontal=True,
        )
        ax.set_yticks(positions)
        ax.set_yticklabels([f"{i + 1}." for i in positions])
        ax.set_xlabel("cumulative steps")
        ax.grid(axis="x")
        ax.set_axisbelow(True)

        for i, action in enumerate(shown):
            ax.annotate(
                action,
                (0.14, i),
                fontsize=8.5,
                va="center",
                ha="left",
                color=pal["text"],
                zorder=6,
                fontfamily="monospace",
            )
        _credit(fig, pal)
    return _finish(fig, path)

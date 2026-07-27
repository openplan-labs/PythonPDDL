"""Command-line interface: ``solve``, ``benchmark``, ``animate`` and ``demo``."""

from __future__ import annotations

import argparse
import os
import sys

from .api import build_task, solve_task, trace_search, validate_plan
from .benchmark import (
    discover_instances,
    plot_summary,
    run_benchmark,
    summarize,
    to_csv,
)
from .heuristics import HEURISTICS
from .search import PLANNERS

INFORMED = {"gbfs", "astar", "wastar", "idastar", "ehc"}


def _add_solve(sub):
    p = sub.add_parser("solve", help="solve a single PDDL instance")
    p.add_argument("domain")
    p.add_argument("problem")
    p.add_argument("-s", "--search", default="astar", choices=sorted(PLANNERS))
    p.add_argument(
        "-H", "--heuristic", default="lmcut", choices=sorted(HEURISTICS) + ["none"]
    )
    p.add_argument(
        "-w", "--weight", type=float, default=2.0, help="weight for weighted A*"
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="watch the search live in the terminal (no dependencies)",
    )
    p.add_argument("--trace", default=None, help="write the search trace as JSON")
    p.add_argument(
        "--plot",
        default=None,
        help="write a four-panel search-progress chart (PNG; needs the viz extra)",
    )
    p.add_argument(
        "--tree", default=None, help="write the radial search-wavefront chart (PNG)"
    )
    p.add_argument(
        "--plan-plot", default=None, help="write the plan timeline chart (PNG)"
    )
    p.add_argument("--dark", action="store_true", help="render charts for dark mode")
    p.add_argument("--quiet", action="store_true", help="do not print the plan")
    p.set_defaults(func=_cmd_solve)


def _add_benchmark(sub):
    p = sub.add_parser("benchmark", help="compare planners over a folder of instances")
    p.add_argument("root", help="folder containing <name>/domain.pddl + problem.pddl")
    p.add_argument("--planners", default="bfs,dijkstra,astar,gbfs,wastar,ehc")
    p.add_argument("--heuristic", default="hff", help="heuristic for informed planners")
    p.add_argument("--csv", default=None, help="write per-run results to this CSV")
    p.add_argument("--plot", default=None, help="write a comparison bar chart (PNG)")
    p.add_argument(
        "--dashboard",
        default=None,
        help="write the full benchmark dashboard: coverage, effort, time, heatmap",
    )
    p.add_argument("--metric", default="expanded")
    p.add_argument("--dark", action="store_true", help="render charts for dark mode")
    p.set_defaults(func=_cmd_benchmark)


def _add_animate(sub):
    p = sub.add_parser("animate", help="replay a search as an animation (MP4 or GIF)")
    p.add_argument("domain")
    p.add_argument("problem")
    p.add_argument("-o", "--output", default="search.mp4")
    p.add_argument("-s", "--search", default="astar", choices=sorted(PLANNERS))
    p.add_argument(
        "-H", "--heuristic", default="lmcut", choices=sorted(HEURISTICS) + ["none"]
    )
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--dark", action="store_true")
    p.set_defaults(func=_cmd_animate)


def _add_demo(sub):
    p = sub.add_parser(
        "demo",
        help="run the bundled demo instances and write every chart to a folder",
    )
    p.add_argument("-o", "--output", default="gallery", help="output folder")
    p.add_argument("--root", default="demos", help="folder of demo instances")
    p.add_argument(
        "--both-modes",
        action="store_true",
        help="render a light and a dark variant of every chart",
    )
    p.add_argument(
        "--animate", action="store_true", help="also render the search animations"
    )
    p.set_defaults(func=_cmd_demo)


# --------------------------------------------------------------------------
def _observers(args):
    """Build the observer for a solve, honouring --live/--trace/--plot."""
    from .trace import MultiObserver, TraceRecorder

    wants_trace = bool(
        args.trace
        or args.plot
        or getattr(args, "tree", None)
        or getattr(args, "plan_plot", None)
    )
    recorder = TraceRecorder() if wants_trace else None
    dashboard = None
    if args.live:
        from .live import TerminalDashboard

        dashboard = TerminalDashboard()
    if recorder and dashboard:
        return MultiObserver(recorder, dashboard), recorder
    return (dashboard or recorder), recorder


def _cmd_solve(args) -> int:
    task = build_task(args.domain, args.problem)
    heuristic = None if args.heuristic == "none" else args.heuristic
    kwargs = {"weight": args.weight} if args.search == "wastar" else {}
    observer, recorder = _observers(args)
    result = solve_task(task, args.search, heuristic, observer=observer, **kwargs)

    trace = recorder.trace if recorder else None
    if trace is not None:
        if args.trace:
            trace.save(args.trace)
            print(f"Wrote trace to {args.trace}")
        _write_charts(args, trace)

    if not result.solved:
        if not args.live:
            print("No plan found.")
        _print_stats(result)
        return 1
    valid = validate_plan(task, result.plan)
    if not args.quiet:
        print(f"Plan ({result.plan_length} steps, cost {result.cost}):")
        for i, op in enumerate(result.plan):
            print(f"  {i + 1:3d}. {op.name}")
    print(f"Valid: {valid}")
    if not args.live:
        _print_stats(result)
    return 0 if valid else 2


def _write_charts(args, trace) -> None:
    targets = [
        (args.plot, "plot_search_progress"),
        (getattr(args, "tree", None), "plot_search_tree"),
        (getattr(args, "plan_plot", None), "plot_plan_timeline"),
    ]
    if not any(path for path, _ in targets):
        return
    try:
        from . import viz
    except ImportError as exc:
        print(f"Charts need the viz extra: {exc}", file=sys.stderr)
        return
    for path, function in targets:
        if not path:
            continue
        getattr(viz, function)(trace, path, dark=args.dark)
        print(f"Wrote {path}")


def _cmd_benchmark(args) -> int:
    instances = discover_instances(args.root)
    if not instances:
        print(f"No instances found under {args.root}", file=sys.stderr)
        return 1
    configs = []
    for planner in args.planners.split(","):
        planner = planner.strip()
        configs.append((planner, args.heuristic if planner in INFORMED else None))

    rows = run_benchmark(instances, configs)
    _print_summary(summarize(rows))
    if args.csv:
        to_csv(rows, args.csv)
        print(f"\nWrote per-run results to {args.csv}")
    if args.plot:
        plot_summary(rows, args.plot, metric=args.metric)
        print(f"Wrote plot to {args.plot}")
    if args.dashboard:
        from .viz import plot_benchmark_dashboard

        plot_benchmark_dashboard(rows, args.dashboard, dark=args.dark)
        print(f"Wrote dashboard to {args.dashboard}")
    return 0


def _cmd_animate(args) -> int:
    from .viz import animate_search

    task = build_task(args.domain, args.problem)
    heuristic = None if args.heuristic == "none" else args.heuristic
    _, trace = trace_search(task, args.search, heuristic)
    animate_search(
        trace, args.output, dark=args.dark, fps=args.fps, seconds=args.seconds
    )
    print(f"Wrote animation to {args.output}")
    return 0


def _cmd_demo(args) -> int:
    """Render the whole gallery: per-instance charts plus a benchmark dashboard."""
    from .viz import (
        plot_benchmark_dashboard,
        plot_plan_timeline,
        plot_planner_comparison,
        plot_search_progress,
        plot_search_tree,
    )

    instances = discover_instances(args.root)
    if not instances:
        print(f"No instances found under {args.root}", file=sys.stderr)
        return 1
    os.makedirs(args.output, exist_ok=True)
    modes = [False, True] if args.both_modes else [False]

    configs = [("astar", "lmcut"), ("astar", "hmax"), ("gbfs", "hff"), ("bfs", None)]
    for instance in instances:
        print(f"-- {instance.name}")
        try:
            task = build_task(instance.domain, instance.problem)
        except Exception as exc:
            print(f"   skipped: {type(exc).__name__}: {exc}")
            continue
        traces = []
        for planner, heuristic in configs:
            try:
                _, trace = trace_search(task, planner, heuristic)
                traces.append(trace)
            except Exception as exc:
                print(f"   {planner}/{heuristic}: {type(exc).__name__}: {exc}")
        if not traces:
            continue
        for dark in modes:
            suffix = "-dark" if dark else ""
            base = os.path.join(args.output, instance.name)
            plot_search_progress(traces[0], f"{base}-progress{suffix}.png", dark=dark)
            plot_search_tree(traces[0], f"{base}-tree{suffix}.png", dark=dark)
            plot_plan_timeline(traces[0], f"{base}-plan{suffix}.png", dark=dark)
            plot_planner_comparison(traces, f"{base}-compare{suffix}.png", dark=dark)
        if args.animate:
            from .viz import animate_search

            animate_search(
                traces[0], os.path.join(args.output, f"{instance.name}-search.mp4")
            )

    rows = run_benchmark(
        instances,
        [
            ("astar", "lmcut"),
            ("astar", "hmax"),
            ("gbfs", "hff"),
            ("ehc", "hff"),
            ("bfs", None),
            ("dijkstra", None),
        ],
    )
    to_csv(rows, os.path.join(args.output, "benchmark.csv"))
    for dark in modes:
        suffix = "-dark" if dark else ""
        plot_benchmark_dashboard(
            rows, os.path.join(args.output, f"benchmark{suffix}.png"), dark=dark
        )
    print(f"\nGallery written to {args.output}/")
    return 0


def _print_stats(result) -> None:
    s = result.stats
    print(
        f"Stats: expanded={s.expanded} generated={s.generated} "
        f"evaluated={s.evaluated} reopened={s.reopened} "
        f"deadends={s.deadends} runtime={s.runtime:.4f}s"
    )


def _print_summary(summary) -> None:
    print(f"{'config':<20}{'coverage':>12}{'expanded':>12}{'runtime(s)':>12}")
    print("-" * 56)
    for key, agg in summary.items():
        cov = f"{agg['coverage']}/{agg['instances']}"
        print(f"{key:<20}{cov:>12}{agg['expanded']:>12}{agg['runtime']:>12.3f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="jupyddl",
        description="Pure-Python PDDL planning framework.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_solve(sub)
    _add_benchmark(sub)
    _add_animate(sub)
    _add_demo(sub)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-07-27

Makes the search **observable**: you can now record it, plot it, watch it live,
animate it, and run the whole thing in a browser.

### Added
- **Search instrumentation** (`jupyddl.trace`): a `SearchObserver` protocol with
  no-op defaults, a `TraceRecorder` that accumulates a serialisable
  `SearchTrace` (JSON round-trip, bounded memory via adaptive thinning), a
  `MultiObserver` for fan-out, and `trace_search()` in the high-level API. Every
  planner emits start/expand/generate/bound/goal/finish events; tracing is
  opt-in and provably transparent (tests assert an observed search returns the
  same plan and statistics as an unobserved one).
- **Zero-dependency live terminal dashboard** (`jupyddl.live`): Unicode
  sparklines, a frontier gauge, live counters and a node rate, repainted with
  ANSI escapes; degrades to periodic one-line output on a non-tty.
- **Visualisation module** (`jupyddl.viz`, the `viz` extra): a validated
  light/dark chart theme plus `plot_search_progress`, `plot_search_tree`
  (radial wavefront), `plot_planner_comparison`, `plot_benchmark_dashboard` and
  `plot_plan_timeline`; `LiveSearchPlot` for notebooks and `animate_search` for
  MP4/GIF replays.
- **Browser playground** (`web/`): runs the unmodified library under Pyodide in
  a web worker, with an editable PDDL editor, live-updating charts, an animated
  search wavefront and a four-planner race. Bundled by `tools/build_web.py` and
  deployed to GitHub Pages.
- **Six demo instances** (`demos/`): gripper, blocksworld8, hanoi, logistics,
  sokoban and elevator, covering type hierarchies, action costs, conditional
  effects, static-predicate pruning and an untyped domain.
- **New CLI commands and flags**: `jupyddl animate`, `jupyddl demo`, and
  `--live`, `--trace`, `--plot`, `--tree`, `--plan-plot`, `--dark` on `solve`;
  `--dashboard` on `benchmark`.
- `tools/make_promo.py`, which renders the project's promo video from numbers it
  measures at render time.

### Changed
- `Planner.search()` and `best_first()` take an optional `observer`; `solve()`
  and `solve_task()` pass one through. Existing calls are unaffected.
- Chart helpers release their figure after writing a file, so rendering a large
  gallery no longer accumulates figures in memory.

## [1.0.0] - 2026-07-01

Complete rewrite: **the Julia dependency is removed** and the project is now a
pure-Python planning framework.

### Added
- Hand-written PDDL tokenizer, AST and recursive-descent parser
  (`jupyddl.parser`).
- Grounder (`jupyddl.grounding`) with type hierarchies, static-predicate
  pruning, positive-normal-form compilation of negative preconditions/goals,
  object harvesting for undeclared constants, and `forall`/`when` conditional
  effect expansion.
- Grounded task representation with conditional effects (`jupyddl.task`).
- Planners (`jupyddl.search`): BFS, DFS, Iterative Deepening, Dijkstra
  (uniform cost), Greedy Best-First, A*, Weighted A*, IDA*, and Enforced Hill
  Climbing, plus a shared best-first engine and a planner registry.
- Heuristics (`jupyddl.heuristics`): blind, goal-count, `h_max`, `h_add`, FF,
  critical-path `h^m` (`h1`/`h2`), and LM-cut, plus a heuristic registry.
- Benchmarking harness (`jupyddl.benchmark`) with CSV export and comparison
  plots, and a CLI (`jupyddl solve` / `jupyddl benchmark`).
- High-level API: `solve`, `build_task`, `solve_task`, `validate_plan`.
- Comprehensive pytest suite covering parsing, grounding, search optimality,
  heuristic admissibility, the API, the benchmark harness and the CLI.

### Changed
- Packaging migrated from `setup.py`/`requirements.txt` to `pyproject.toml`
  (hatchling); the core has **zero runtime dependencies** (matplotlib is an
  optional `viz` extra).
- CI reworked to run on modern Python without Julia.

### Removed
- The Julia / `PDDL.jl` / PyCall / pyjulia integration and the old
  `AutomatedPlanner` / `DataAnalyst` API.

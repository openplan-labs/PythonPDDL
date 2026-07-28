# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2026-07-28

Closes the PDDL gap: 20 of the 21 requirement flags are now supported, up from
15. Everything new is a source-to-source compilation applied before grounding,
so the search engine is untouched.

### Added
- **`jupyddl.compile`**: PDDL 3 constructs are rewritten into the classical core
  before grounding. Every fact and action it invents is prefixed `__` and hidden
  from printed plans.
- **`:constraints`** — `always`, `at-end`, `sometime`, `sometime-before`,
  `sometime-after` and `at-most-once`. `always` becomes a precondition on every
  action plus a goal conjunct (the three checks between them cover every state
  on a trajectory). `sometime`/`sometime-before` use optional monitor actions;
  `sometime-after`/`at-most-once` use *forced* monitors on conditional effects,
  because a constraint the planner could satisfy by not looking would not be a
  constraint. The metric-time forms (`within`, `hold-during`, ...) are refused
  by name.
- **`:preferences`** — goal preferences become a priced choice: a closing action
  freezes the state, then each preference is resolved for free if it holds or at
  its `(is-violated p)` weight if not, so plain cost-optimal search minimises the
  metric. Freezing first is what stops a preference being satisfied halfway
  through and then broken. Soft trajectory constraints are refused.
- **`:timed-initial-literals`** — elapsed time becomes a numeric fluent advanced
  by action durations, with a firing action per literal, a wait action that
  advances the clock, guards that stop anything else happening while a due
  literal has not fired, and an ordering constraint so literals fire in time
  order.
- **`:object-fluents`** — compiled to a predicate plus a uniqueness rule;
  `assign` clears the old value first so the function stays single-valued.
  Nested-term use is refused in favour of the equality form.
- **`:duration-inequalities`** — bounds are collected and the shortest feasible
  duration chosen, which is makespan-optimal with no concurrency. Strict
  `<`/`>` are refused: they have no shortest feasible value.
- New demos: `errands` (preferences + constraints) and `timed-market` (timed
  literals).

### Changed
- The promo video is re-cut for what the library became: three new scenes cover
  the 21-flag support matrix, the soft-goal trade-off in `errands` and the
  timed-literal schedule in `timed-market`, and the browser scene now shows the
  four-view workbench. Still 100% measured at render time.
- `Task.makespan` replays the clock when a task has one. Waiting for a timed
  literal advances time without any action taking that long, so summing
  durations under-reported the end time.
- `:fluents` now genuinely covers both halves; `:preferences`, `:constraints`,
  `:timed-initial-literals`, `:object-fluents` and `:duration-inequalities`
  moved off the rejected list.

### Still not supported
- `:continuous-effects`, and **true temporal concurrency**. Durative actions
  compile to a sequential schedule and never overlap, so a plan needing two
  actions to run at once will not be found. That needs a mutex-aware temporal
  scheduler rather than another compilation.

## [2.1.0] - 2026-07-28

Grows the PDDL fragment from STRIPS to most of PDDL 3.1, and turns the browser
playground into a research workbench.

### Added
- **`jupyddl.requirements`**: one registry saying what happens to every PDDL
  requirement flag — native, compiled, partial or rejected — with the reason.
  The parser, `jupyddl requirements`, the README table and the web UI all read
  it, so they cannot drift apart.
- **Full ADL conditions**: `or`, `imply`, `exists` and nested `not` in
  preconditions, goals and `when` conditions. The parser produces negation
  normal form; the grounder expands quantifiers over the object pool and
  distributes to DNF, emitting one operator per disjunct. A disjunctive goal is
  compiled to an artificial goal fact reached by a zero-cost operator per
  disjunct, hidden from the printed plan via `Task.visible_plan`.
- **Derived predicates** (`:derived`): rules are grounded into `Axiom`s and
  closed to a least fixpoint after every state change, and enter the
  delete-relaxation as zero-cost rules so the heuristics stay admissible.
- **Numeric fluents**: comparisons in preconditions and goals, and
  `assign`/`increase`/`decrease`/`scale-up`/`scale-down` effects over arithmetic
  expressions. Numeric tasks carry a `State` with a value vector; classical
  tasks keep the bare frozenset and pay nothing.
- **Durative actions**: parsed with `at start`/`over all`/`at end` conditions and
  effects, compiled to sequential actions carrying their duration, with
  `Task.makespan`. Concurrency is explicitly *not* modelled — see the support
  table.
- **Five more planners**: hill climbing (`hc`), beam search (`beam`), Iterated
  Width (`iw`), branch and bound (`bnb`) and anytime weighted A* (`awastar`).
- **Search budgets**: `max_expansions` and `time_limit` on every planner, the
  API and the benchmark harness. A run that stops on its budget sets
  `stats.truncated`, so an empty result never masquerades as a proof of
  unsolvability.
- **`jupyddl.generator`**: seven reproducible instance generators — same
  *(kind, size, seed)*, same bytes — plus `jupyddl generate`.
- **New demo instances**: `rovers` (ADL), `network` (recursive axioms),
  `numeric-transport`, `workshop` (temporal) and `blocksworld12`.
- **New CLI commands**: `jupyddl requirements` and `jupyddl generate`;
  `--max-expansions` / `--time-limit` on `solve` and `benchmark`.
- **The web workbench**: four views — Solve, Experiment (an instance ×
  configuration sweep with a sortable table and CSV/JSON export), PDDL support,
  and Generate.

### Changed
- Planners take their initial state from `Task.initial_state()` and step through
  `Task.apply()`, so axioms close and numeric values flow without every planner
  knowing about either.
- `validate_plan` replays through the task instead of the raw operators; it
  previously started from `task.init` and so saw neither axioms nor numbers.
- `Operator.base_name` strips compilation tags, so a plan prints the action the
  domain author wrote rather than `move(a,b)#2`.
- `SearchResult.plan_names(canonical=True)` returns those canonical names.
- Budget clock checks happen every expansion, not every 256: under LM-cut a
  single expansion can cost tens of milliseconds, and the coarse interval
  overshot a short time limit by more than an order of magnitude.

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

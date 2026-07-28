# AGENTS.md

## Cursor Cloud specific instructions

`jupyddl` is a **pure-Python** PDDL planning framework (parser, grounder,
planners, heuristics, instrumentation, visualisation, benchmarking). The
Julia/`PDDL.jl`/PyCall integration has been removed — there is no Julia, no
native build step, and the core has zero runtime dependencies.

### Environment
- Python ≥ 3.9; a `.venv` (created with `uv`, Python 3.12) with an editable
  install: `uv pip install -e ".[dev]"` (add `viz` for the matplotlib charts,
  animations and benchmark dashboards). `.venv`, `web/vendor/` and
  `pddl-examples/` contents are git-ignored.
- The `pddl-examples` git submodule supplies the small domains/problems the
  parser and grounder tests use; it must be initialised
  (`git submodule update --init`). The larger instances in `demos/` live in this
  repository and are always available.

### Running / testing / linting (use the venv interpreter)
- Tests: `.venv/bin/python -m pytest` (add `--cov=jupyddl`).
- Lint (as CI): `flake8 jupyddl tests tools` (config in `.flake8`, max-line 100).
  Format with `black jupyddl tests tools`.
- CLI: `.venv/bin/python -m jupyddl.cli solve <domain> <problem> -s astar -H lmcut`
  or `... benchmark demos --dashboard out.png`. Installed as `jupyddl` too.
  Also `jupyddl animate` (MP4/GIF replay) and `jupyddl demo` (full chart gallery).

### Layout beyond the core
- `jupyddl/requirements.py` — **the source of truth** for what every PDDL
  requirement flag does here. Change support for a feature *here first*; the
  parser, the CLI, the README table and the web UI all read it.
- `jupyddl/generator.py` — reproducible instance generators.
- `jupyddl/trace.py` — search observers, events, `SearchTrace` (JSON).
- `jupyddl/live.py` — the terminal dashboard; **stdlib only, keep it that way**,
  it is what makes "watch a search" free of dependencies.
- `jupyddl/viz/` — everything that imports matplotlib. Nothing in the core may
  import this package.
- `web/` — the Pyodide playground; `tools/build_web.py` bundles the package
  sources and demos into `web/dist` (committed).
- `tools/make_promo.py` — renders the promo video from measured runs.

### The condition pipeline
Conditions are a **formula tree in negation normal form**: `parse_condition`
pushes every `not` down to the literals and rewrites `imply`, so nothing
downstream sees a negated compound. Quantifiers survive parsing because
expanding them needs the object pool; `grounding._dnf` expands them and
distributes to DNF, and each disjunct becomes its own operator (named
`action(args)#N`). `Operator.base_name` strips that tag for display.

### Optional task layers
`Task` carries three layers that are inert unless the domain uses them, and the
planners must go through the task rather than the operator to honour them:

- **axioms** — `Task.apply()` re-closes derived predicates after every step, and
  `Task.initial_state()` closes the initial state. A planner that calls
  `op.apply(state)` directly will silently skip this.
- **numeric fluents** — the state becomes a `State(facts, values)` instead of a
  frozenset. Anything doing `set(state)` still works (`State` is iterable); use
  `facts_of(state)` when you need the fact set specifically.
- **durations** — `op.duration` plus `Task.makespan(plan)`.

### Non-obvious notes
- **Instrumentation must stay transparent.** Planners only touch the observer
  behind `if observer is not None`, so the default path keeps its zero-overhead
  behaviour. `tests/test_trace.py` asserts that an observed search returns the
  same plan and statistics as an unobserved one — do not break that.
- **`web/dist` is generated and committed.** After changing anything under
  `jupyddl/` (including a `black` reformat) run `python tools/build_web.py` and
  commit the result, or `tests/test_web_bundle.py` and the Pages workflow fail.
  The bundle deliberately excludes `jupyddl/viz/` (matplotlib is not loaded in
  the browser).
- **The playground's Python lives in `web/bootstrap.py`**, fetched at runtime
  rather than embedded in `worker.js`. Do not inline it back into a JS template
  literal: reStructuredText double-backticks in a docstring terminate the
  template and break the worker.
- **Example data quirks (external submodule, do not "fix" in this repo):**
  `grid` uses numeric fluents and is intentionally unsupported (raises
  `UnsupportedFeatureError`); `vehicle` has typos in its problem file
  (`struck`/`truck`, `acessible`) so its goal is unreachable and it is correctly
  reported unsolvable. Tests treat both as expected.
- **Conditional effects (`flip`, `elevator`)**: the delete-relaxation heuristics
  (`hadd`, `hff`, `lmcut`, `h^m`) are *not guaranteed admissible* on domains with
  conditional effects because each conditional effect is relaxed into its own
  operator. For guaranteed-optimal plans there use `bfs`, `dijkstra`, or
  `astar`/`idastar` with the `blind` heuristic. Optimality tests use these.
- **Matplotlib is optional**: only `jupyddl.viz` and `jupyddl.benchmark.plot_summary`
  need it; run headless with `MPLBACKEND=Agg` if no display (`jupyddl.viz` already
  forces the Agg backend).
- **Budgets are cooperative.** Python cannot safely interrupt a running search,
  so planners poll a `Budget` in their main loop. If you add a planner with an
  unbounded loop, poll it — `bnb` and `iw` will otherwise run for hours. A run
  stopped by a budget sets `stats.truncated`; never report such a run as
  unsolvable.
- **The clock is checked every expansion** (`Budget.check_every=1`). That looks
  wasteful but is not: one LM-cut expansion can cost tens of milliseconds, and a
  coarser interval overshoots a short `--time-limit` enormously.
- Extend via the registries: `jupyddl.search.PLANNERS`,
  `jupyddl.heuristics.HEURISTICS` and `jupyddl.generator.GENERATORS`. The CLI,
  benchmark harness and web workbench all read from them, so a new entry shows
  up everywhere for free.

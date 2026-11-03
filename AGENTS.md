# AGENTS.md

## Cursor Cloud specific instructions

`jupyddl` is a **pure-Python** PDDL planning framework (parser, grounder,
planners, heuristics, instrumentation, visualisation, benchmarking). The
Julia/`PDDL.jl`/PyCall integration has been removed — there is no Julia, no
native build step, and the core has zero runtime dependencies.

### Environment
- Python ≥ 3.10; a `.venv` (created with `uv`, Python 3.12) with an editable
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

### What CI actually checks
- **`tests`** — lint (`flake8 jupyddl tests tools`) and the suite on Python
  3.10–3.14. Most of the matrix installs `.[dev]` *without* matplotlib, which is
  what nearly everyone installs and keeps the core honest about not importing
  it; one entry installs `.[dev,viz]` so `jupyddl/viz/` is executed rather than
  skipped. `tests/test_viz.py` and `tests/test_cli_viz.py` `importorskip`
  matplotlib — 30 tests with it, 4 without — so that one entry is the only
  thing standing between the charting surface and no coverage at all.
- **`build`** — builds the sdist and wheel, `twine check`s them, installs the
  wheel into a clean venv and plans with it from outside the repository. It
  deliberately does *not* repeat the lint: it is there to catch packaging
  breakage (a module missing from the wheel, an entry point that does not
  resolve), which an editable install hides.
- **`format`** — runs `black` on a push to main and commits the result. It
  rebuilds `web/dist` in the same commit; reformatting a bundled source and
  committing without the rebuild would leave main in a state where `pages`
  refuses to deploy.
- **`release`** — on every push to main, `plan` reads the merged PR (title,
  its commits, and every commit since the last `v*` tag) and picks the bump
  (`tools/release.py`: `fix`/`perf` patch, `feat` minor, `!`/`BREAKING CHANGE`
  major; a `release:*` label overrides). `bump` then edits the version in
  `pyproject.toml`, `jupyddl/__init__.py` and `CITATION.cff`, dates
  `[Unreleased]` in the changelog, rebuilds `web/dist`, commits
  `chore(release): X.Y.Z` to main, tags it and starts `pages`. No release PR
  and no approval: the workflow token may push but not open PRs, so this is
  the shape that needs no extra settings. **PR titles must be Conventional
  Commits** (`pr-title` checks). Also fires on a hand-pushed `v*` tag. Builds, refuses a tag that disagrees
  with `pyproject.toml` or a `__version__` that disagrees with either, runs
  `twine check --strict`, installs the wheel clean and plans with it, publishes
  to PyPI over OIDC (no stored token), then cuts the GitHub Release from the
  changelog section for that version. `.docs/RELEASING.md` is the runbook,
  including the one-time PyPI trusted-publisher setup only a maintainer can do.
  Do not bump versions by hand: merging a `feat:`/`fix:` PR does it in all
  three places and rebuilds `web/dist`, which carries it too.
- **`pages`** — bundles, refuses to deploy a stale `web/dist`, then deploys.
  Its job is called `bundle`, not `build`: a check named `build` has to mean
  the packaging workflow, and two checks with one name is a reliable way to gate
  a merge on the wrong one. (The repository's Mergify rules used to key on it.)
- **Triggers.** `tests` and `build` run on pull requests, so a contributor's
  fork gets CI, and on pushes to `main`. A branch pushed without a PR gets no
  CI; open a draft PR.

### Layout beyond the core
- `jupyddl/requirements.py` — **the source of truth** for what every PDDL
  requirement flag does here. Change support for a feature *here first*; the
  parser, the CLI, the README table and the web UI all read it.
- `jupyddl/compile.py` — PDDL 3 (preferences, trajectory constraints, timed
  initial literals, object fluents) rewritten into the classical core **before**
  grounding. Add front-end features here as source-to-source transformations,
  not as special cases in the grounder or the search.
- `jupyddl/learn/` — learned heuristics. **Stdlib-only like the core**; NumPy is
  a speed option and the pure-Python path must keep working. Two stages:
  `train.py` imitates `h*` from solved plans, `rl.py` optimises search cost
  directly. Nothing in the core imports it — `jupyddl.heuristics` resolves
  `learned:<path>` lazily inside the loader, so a planner that never asks for
  one never pays for it.
- `jupyddl/generator.py` — reproducible instance generators.
- `jupyddl/trace.py` — search observers, events, `SearchTrace` (JSON).
- `jupyddl/live.py` — the terminal dashboard; **stdlib only, keep it that way**,
  it is what makes "watch a search" free of dependencies.
- `jupyddl/viz/` — everything that imports matplotlib. Nothing in the core may
  import this package.
- `web/` — the Pyodide playground; `tools/build_web.py` bundles the package
  sources and demos into `web/dist` (committed). It also writes
  `capabilities.json` (the registries) and `research.json` (distilled from
  `.docs/assets/rl-data.json`, so the page and the RL video quote one measured run).
  Those two are rendered **before** Pyodide loads — the app shell is never
  hidden, and only the run controls are gated on `state.ready` — so a stale
  bundle briefly states something untrue rather than merely lagging.
  `tests/test_web_bundle.py` pins both.
- `tools/make_promo.py` — renders the main promo video from measured runs.
- `tools/make_learn_promo.py` — the learned-heuristic/RL video. It re-measures
  everything including both failure modes, so it cannot drift from `.docs/`;
  `.docs/assets/rl-data.json` caches the pass, delete it to re-measure.
- **`.docs/` is the only documentation directory.** Research notes and the
  release runbook sit at its top level; every image, video and measurement
  cache goes in `.docs/assets/`. There is no `docs/` or `promo/` — they were
  merged in because three directories with no rule between them meant every
  new file was a guess. The two exceptions live in `.github/`:
  `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`, which GitHub only recognises in
  the root, `.github/` or `docs/`.
  The sdist excludes `.docs/assets/*.png` and `*.mp4` **by extension, not by
  directory**, because `rl-data.json` sits beside them and
  `tests/test_web_bundle.py` reads it — excluding the directory ships an sdist
  whose own suite fails.

### The condition pipeline
Conditions are a **formula tree in negation normal form**: `parse_condition`
pushes every `not` down to the literals and rewrites `imply`, so nothing
downstream sees a negated compound. Quantifiers survive parsing because
expanding them needs the object pool; `grounding._dnf` expands them and
distributes to DNF, and each disjunct becomes its own operator (named
`action(args)#N`). `Operator.base_name` strips that tag for display.

### The PDDL 3 compilations
Everything `compile.py` introduces is named with a leading `__`, and grounding
collects those into `Task.synthetic` so `Task.visible_plan` can hide them. Two
traps worth knowing:

- **Synthetic actions must declare a zero cost.** Without a `(total-cost)`
  metric, grounding charges 1 for any action with no cost effect (under one, 0,
  as IPC planners do), which would silently inflate the metric — closing a
  plan or observing a constraint is bookkeeping, not work.
- **Forced vs optional monitors.** `sometime` can use an action the planner
  applies when convenient; `at-most-once` and `sometime-after` cannot, because
  the planner would simply decline to notice. Those ride on conditional effects
  added to every domain action.

- **Durative actions are one sequential action** with three time points:
  at-start conditions in the starting state, then at-start effects, then
  over-all/at-end conditions (and `always`) checked in that in-between state,
  then at-end effects, which win any conflict (`grounding._sequence`). What
  cannot be expressed that way is refused by name, not approximated.
- **The cost model.** No metric: every action costs 1 (or its declared
  `total-cost` increase). A metric over `(total-cost)`: declared increases,
  0 otherwise; increases may read *static* functions (`road-length`), which
  are evaluated at grounding. A metric without `total-cost`: domain actions
  are free and only preference penalties count. `(total-time)`: an action
  costs its duration. `maximize` and other metric shapes are refused.
- `compile_problem` deep-copies its inputs: `ground()` must never mutate a
  parsed Domain/Problem, which callers may reuse for another problem.

### Optional task layers
`Task` carries three layers that are inert unless the domain uses them, and the
planners must go through the task rather than the operator to honour them:

- **axioms** — `Task.apply()` re-closes derived predicates after every step, and
  `Task.initial_state()` closes the initial state. A planner that calls
  `op.apply(state)` directly will silently skip this.
- **numeric fluents** — the state becomes a `State(facts, values)` instead of a
  frozenset. Anything doing `set(state)` still works (`State` is iterable); use
  `facts_of(state)` when you need the fact set specifically.
- **durations** — `op.duration` plus `Task.makespan(plan)`. When timed initial
  literals introduced a clock, `makespan` replays it instead of summing
  durations: waiting advances time without any action taking that long.

### Non-obvious notes
- **Instrumentation must stay transparent.** Planners only touch the observer
  behind `if observer is not None`, so the default path keeps its zero-overhead
  behaviour. `tests/test_trace.py` asserts that an observed search returns the
  same plan and statistics as an unobserved one — do not break that.
- **`web/dist` is generated and committed.** After changing anything under
  `jupyddl/` (including a `black` reformat) run `python tools/build_web.py` and
  commit the result, or `tests/test_web_bundle.py` and the Pages workflow fail.
  The bundle deliberately excludes `jupyddl/viz/` (matplotlib is not loaded in
  the browser). The builder must stay **byte-reproducible**: it sorts both the
  directory walk and the JSON keys, because an unsorted `os.walk` bundles the
  same sources in a disk-dependent order and the staleness check then fails on
  a bundle that is not stale.
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
  operator. `lmcut` and `h2` clear their `admissible` flag on such tasks. For
  guaranteed-optimal plans there use `dijkstra`, or `astar`/`idastar` with the
  `blind` heuristic. Optimality tests use these.
- **Fact ids are assigned in sorted order** and every id set is built sorted
  (`grounding._ids`). A set of ints iterates in insertion order where ids share
  a hash slot, so filling one while walking atoms in hash order made LM-cut's
  tie-breaks — and so expansions and plans — depend on `PYTHONHASHSEED`.
  `tests/test_semantics.py` runs a search under three seeds to pin this.
- **`validate_plan` replays through `Task.apply`**, so it cannot catch a
  semantics bug in the task itself. `tests/test_semantics.py` pins semantics
  on tiny hand-written tasks whose answer is obvious by inspection — add a case
  there when you change grounding, axioms, complements or costs.
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
### Learned heuristics, non-obvious parts
- **Features are keyed on the predicate symbol, never the ground atom**, and
  normalised per symbol. That is what makes the vector the same length for 4
  blocks and 40. Getting `predicate_of` wrong does not raise — it gives every
  ground atom its own slot and silently destroys transfer, so
  `tests/test_learn.py` pins both spellings (`(on a b)` and `move(a,b)`).
- **The default objective is ranking, not regression.** GBFS reads the *order*
  a heuristic imposes, never its values; a model uniformly 30 too high guides
  perfectly. Checkpoint selection is on top-1 accuracy for the same reason. The
  regression term is kept at a small weight only to anchor a scale, which a
  pure ranking loss leaves undefined and `wastar` needs.
- **Report the distribution, not just the mean.** The held-out blocksworld set
  has a heavy tail: nine of ten instances land between 58 and 227 expansions and
  the tenth is worth thousands, so the mean is close to a report of that one
  instance. Two published claims here were wrong because of it — see the
  correction in `.docs/rl-for-search.md`, which also records the more
  embarrassing cause: two settings changed in one edit and the improvement was
  credited to the wrong one.
- **Two things decide whether the RL stage does anything.** It must start from
  the imitation solution (search cost is flat over every parameter vector that
  solves nothing), and it must tune on instances with *headroom* — on the
  training ladder the heuristic already expands about as many nodes as the plan
  is long, so every perturbation scores the same. Measured: tuning on the
  training sizes moved the score 12.83 → 12.75; tuning a rung higher moved it
  1605 → 64. `learn_heuristic` defaults `cem_sizes` above the training ladder.
- **A learned heuristic is never admissible** and must not be used to claim an
  optimal plan. The suite asserts plans stay *valid*, which is the invariant
  that does hold.
- Extend via the registries: `jupyddl.search.PLANNERS`,
  `jupyddl.heuristics.HEURISTICS` and `jupyddl.generator.GENERATORS`. The CLI,
  benchmark harness and web workbench all read from them, so a new entry shows
  up everywhere for free. Other packages reach the same registries through
  entry points (`jupyddl/plugins.py`). Discovery is **lazy** — the CLI runs it
  before building its parser, the library only on a lookup miss — so importing
  jupyddl never scans installed packages, and `INFORMED_PLANNERS` /
  `OPTIMAL_PLANNERS` describe the built-ins only. The suite sets
  `JUPYDDL_NO_PLUGINS=1` so a developer's installed plugins cannot change it.
- **`.docs/EXTENDING.md` is executed** by `tests/test_extending.py`: change an
  API it uses and that test fails until the guide is updated too.

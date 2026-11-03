# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.5.0](https://github.com/openplan-labs/PythonPDDL/compare/v2.4.0...v2.5.0) (2026-11-03)


### Features

* automate semantic releases with release-please ([#169](https://github.com/openplan-labs/PythonPDDL/issues/169)) ([9d05b56](https://github.com/openplan-labs/PythonPDDL/commit/9d05b56a4cc3a514f992c0cc0c5be1b218d2a5e2))


### Bug Fixes

* **learn:** honest ranking metrics, seed-sensitive generators, distinct benchmark configs ([#171](https://github.com/openplan-labs/PythonPDDL/issues/171)) ([e6ba76f](https://github.com/openplan-labs/PythonPDDL/commit/e6ba76f5c3ca177d04dd9de06d85332601f2e098))
* **pddl:** correct durative, PDDL 3 and metric semantics; refuse what cannot be honoured ([#172](https://github.com/openplan-labs/PythonPDDL/issues/172)) ([3646e55](https://github.com/openplan-labs/PythonPDDL/commit/3646e55a12bb3b089d4d48838d5e0d88c2d03dfd))

## [Unreleased]

### Added
- **Semantic releases, decided by what was merged.** Pull request titles are
  now Conventional Commits (a `pr-title` check enforces it), and
  release-please keeps one release PR whose version follows from them:
  `fix` → patch, `feat` → minor, `feat!` → major. Merging that PR tags the
  commit and starts the existing PyPI pipeline. The hand-written notes in this
  file are filed under the new version automatically, and `web/dist` is
  rebuilt with the new version. See `.docs/RELEASING.md`.
- **The intent to publish, written down.** `.docs/PUBLISHING.md` says why and
  where we mean to publish jupyddl as citable research software — a Zenodo DOI
  per release, a JOSS paper, ICAPS workshops for the research results — with a
  readiness checklist, the paper outline, the AI-usage disclosure JOSS requires
  and how contributors can be considered for authorship. The README states the
  project's aim (a framework to build on and to explain algorithms with) and
  its Cite section says which version the 2021 preprint describes.
- **Plugins.** A package can now add planners, heuristics and generators by
  declaring `jupyddl.planners`, `jupyddl.heuristics` or `jupyddl.generators`
  entry points; once it is installed, `jupyddl solve -s <its planner>` works
  without anyone importing it. Discovery is lazy (the CLI runs it before
  building its parser, the library only when asked for a name it does not
  know), a built-in name always wins, a plugin that fails to import is skipped
  with a warning, and `JUPYDDL_NO_PLUGINS=1` turns it off. `jupyddl plugins`
  lists what was found.
- **`.docs/EXTENDING.md`, a guide to building on the framework**: a heuristic,
  a planner from a priority function, a planner from scratch with the full
  budget/observer contract, a generator, how to test the guarantees you claim,
  and how to ship it all as a plugin. Every example runs in the test suite, so
  the guide cannot drift from the API.
- `best_first` accepts a **tuple priority** for tie-breaking
  (`lambda g, h: (g + h, h)`); observers are given its first element as `f`.
- **Community files for outside contributors**: a security policy, a
  pull-request template, a *Propose an algorithm* issue template, and an
  "Adding an algorithm" section in `CONTRIBUTING.md`.

### Changed
- **CI now runs on pull requests, including from forks.** `tests` and `build`
  were `on: [push]`, which never fires for a fork, so an outside contributor's
  PR showed no checks at all. They now run on every pull request and on pushes
  to `main`.
- **Documentation lives in one place: `.docs/`.** It was spread over three
  directories — `docs/` for the contributor files and the README charts,
  `promo/` for the videos and screenshots, `.docs/` for the research notes —
  with no rule saying which took what, so every new file was a guess. The
  research notes, the release runbook and every image, video and measurement
  cache are now under `.docs/`, with binaries in `.docs/assets/`. This also
  matches `pymapf`, the sibling project.

  Paths quoted in the released sections below are the ones that were correct
  at the time. `promo/rl-data.json` is now `.docs/assets/rl-data.json`, and
  `docs/images/` is now `.docs/assets/`.

  `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` moved to `.github/` rather than
  `.docs/`: GitHub only recognises those two files in the repository root,
  `.github/` or `docs/`, so filing them under `.docs/` would have silently
  dropped the contributing link on the issue and pull-request forms.

### Fixed
Wrong answers from the task model and the heuristics. Each of these returned a
plan, a cost or a value without complaint, and `validate_plan` accepted the
plans because it replays them through the same model. They are now pinned by
`tests/test_semantics.py` on tiny tasks whose right answer is obvious.

- **Derived facts were never retracted.** An action that deleted the support of
  a derived fact left it true, and `(not <derived>)` held from the initial state
  on. Axioms are now recomputed from scratch after every step, and a negated
  derived fact is maintained with them. Negating a derived predicate *inside
  another rule* needs stratification and is now refused by name.
- **An atom both added and deleted left its complement true as well**, so
  `(not (at a))` and `(at a)` could hold together, and a `when` that re-added a
  deleted atom did the same. For complement facts the delete now wins.
- **Action costs.** Under a `(total-cost)` metric an action with no cost effect
  now costs 0, as in IPC planners, rather than 1 — on pegsol and sokoban the
  "optimal" planners were optimising the wrong objective. A cost inside `when`
  was charged whether or not the condition held; it is now refused, as is a
  negative cost.
- **A numeric fluent that is never defined read as 0**, so `(distance a c)` for
  a road that does not exist cost nothing. Conditions and effects reading one
  now fail, as PDDL 2.1 specifies.
- **`blind` was not admissible** when any action is free, which every PDDL 3
  compilation introduces, so `astar`+`blind` could return a costlier plan.
- **`h1`/`h2`/`hm` crashed on every numeric or timed task**, and **LM-cut
  crashed on an empty goal**, which made the default `solve()` crash too.
- **`lmcut` and `h2` claimed admissibility on tasks with conditional effects**,
  where they can overestimate. The instance flag now says so.
- **`solve_task(task, "astar")` used `hff`**, which is not admissible, so the
  default A* was not optimal. Optimal planners now default to `lmcut`. `-H none`
  also fell through to `hff`; it now means `blind`.
- **`make_heuristic(LMCutHeuristic, task)` returned the class unbuilt**, and the
  planner then crashed calling it on a state. Classes and partials are now built.
- **Results depended on `PYTHONHASHSEED`.** Fact ids followed string hashing,
  so LM-cut broke ties differently in every process: gripper with `astar`/`lmcut`
  expanded 1725, 1758 or 1760 nodes and returned different plans. Ids are now
  assigned in sorted order, so a benchmark reproduces.
- **`iddfs`, `idastar` and `ehc` ignored their budget**, so `--time-limit 0.5`
  on blocksworld12 ran for minutes. `beam` polled once per layer and `bnb` kept
  evaluating LM-cut on every sibling after the budget fired; both now stop at
  once. A new contract test runs every planner where the budget is the only
  thing that stops it.
- **A depth cap was reported as "no plan exists"** (`iddfs`, `bnb`), and a
  `bnb` plan found under one as optimal. Hitting the cap now sets `truncated`,
  as does a search deeper than Python's recursion limit.
- **`bfs` and `iddfs` were listed as optimal.** They minimise plan *length*;
  on errands they return cost 16 where 10 is optimal. They are no longer in
  `OPTIMAL_PLANNERS`, and the workbench no longer badges them.
- **The live dashboard** said "no plan found" when the budget stopped the
  search, smeared every repaint on a narrow terminal (lines were never clipped
  to its width), and logged ten times too often when not on a terminal.
- `idastar`, `ehc`, `hc` and `beam` without a heuristic raise a clear
  `ValueError` instead of a `TypeError` from deep inside the search;
  `MultiObserver` accepts positional arguments like the base class; the
  `SearchObserver` docstring no longer claims hooks are called defensively.
- **The command line's exit codes say what happened**: 0 solved, 1 no plan, 2
  bad input, 3 stopped by a budget, 4 a plan that failed validation. Before,
  unsolvable, out-of-budget and a crash all exited 1. Bad input — a missing
  file, a parse error, an unknown planner — is one line on stderr instead of a
  traceback (`JUPYDDL_DEBUG=1` brings the traceback back). `--version` exists.
- **`jupyddl demo` and `jupyddl benchmark demos` never finished**: neither took
  a budget, and `demo`'s blind runs on blocksworld12 do not end. Both now stop
  each search after 10 s by default; `animate` and `demo` accept
  `--max-expansions` and `--time-limit` like `solve`.
- `benchmark` checks `--planners`, `--heuristic` and `--metric` before running
  rather than reporting `nosuch 0/13`, and every chart option checks for the
  `viz` extra *before* searching instead of failing (or silently writing
  nothing) afterwards.
- **A dead end crashed the charts** (`Axis limits cannot be NaN or Inf`) and
  coloured the wavefront NaN; in the workbench, an infinite `h` or `f` made the
  progress message invalid JSON and aborted the run.
- **A trace's plan showed compiler bookkeeping** (`__observe-1#1`) and
  disjunct tags; it now records the plan the user reads.
- **The sdist's own test suite failed**: it dropped `.github/` entirely while a
  test checks `CONTRIBUTING.md` is there. Only `.github/workflows/` is left
  out now. `build` runs `twine check --strict`, like `release` does.
- `black` is pinned to one major version in both the dev extra and the
  `format` workflow, which installed whatever was newest; Renovate now pins
  third-party actions to commit digests.
- **Learned-heuristic top-1 was neither held out nor tie-safe.** A sibling
  scoring level with the successor the plan took counted as a correct ranking,
  so a model that outputs a constant scored 1.000 — and checkpoint selection,
  which reads top-1, could prefer a collapsed epoch. `Corpus.split` also gave
  the held-out side no ranking groups, so the "held out" top-1 `jupyddl learn`
  printed was measured over training instances. A tie with `k` siblings now
  earns `1/(k+1)`, a sibling identical to the chosen successor in features and
  step cost is not recorded at all, and selection and the reported top-1 use
  the held-out groups. The published figures came from the old rule; see the
  dated corrections in `.docs/learned-heuristics.md` and
  `.docs/rl-for-search.md`.
- **A DAgger or bootstrap state could validate against its own instance.**
  `split` treated `X#dagger1` as an instance apart from `X`; it now groups on
  the name before the `#`.
- **`summarise_transfer` averaged each heuristic over the runs it solved**, so
  one that solved only the easy instance looked many times better than one that
  solved both. An unsolved run is now charged twice the budget, as
  `search_cost` charges it, and plan cost compares only instances every
  configuration with any coverage solved. The imitation-only blocksworld mean
  the docs quote as 366 is 6 329 under that rule.
- **A malformed learned heuristic loaded, then gave one answer with NumPy and
  another without.** A network whose input width is not the feature vector's,
  a ragged row, a `sizes` field that does not describe the weights, a
  non-finite value or a non-positive scale is now a `ValueError` on load.
  `MLP.set_flat` checks the length before writing anything, instead of leaving
  a wrong-length vector half-applied.
- **Every negated fact shared one feature, `not`.** Complement facts now key on
  the predicate they negate (`not-sampled`), so rovers and workshop get a slot
  per negated predicate. A model trained on a domain with negative
  preconditions before this sees different features now; retrain it.
- `train()` with a pure ranking objective and no ranking groups returned its
  random initialisation as a trained model; it now raises. `learn_heuristic`
  spent a generator passed as `sizes` on its first read, recording `[]` and
  raising under `cem_iterations`; size arguments are now read once, on entry.
- **`gripper` and `workshop` ignored their seed.** Twenty seeds gave one
  instance, so a split by seed held out copies of training instances, and the
  RL stage's tuning and validation sets were the same problems. The seed now
  decides which room each ball and the robot start in (every ball still has to
  change rooms), and how far along the pipeline each part starts. **Every
  generated gripper and workshop instance changes**; `demos/workshop` is the one
  in which every part starts raw, and `generate` no longer reproduces it.
- **Generators could emit an instance solved before it starts.** `blocksworld`
  compared tower lists in order and skipped the check for two blocks, where 75
  of the first 200 seeds came out trivial; `random-strips` could draw a goal
  that already held. Both now re-draw only in that case, so every instance that
  was not trivial is byte-identical to before.
- **The benchmark harness merged configurations that differed in planner
  arguments** — `wastar` at weights 1 and 5 summarised as one entry with twice
  the instances — and recorded `("gbfs", None)` as heuristic `""` although it
  ran `hff`. Rows carry a `config` label with the arguments
  (`wastar/hff(weight=5.0)`, also a new CSV column), which `summarize` and the
  dashboard group on; a repeated config is suffixed `#2`; and `heuristic` is
  what `solve_task` actually ran, worked out by the new
  `jupyddl.api.resolve_heuristic`.

Wrong answers from the PDDL front end. The parser and the PDDL 3 compilations
handed the grounder a different problem from the one written, so plans
validated — against the wrong problem. Each case is now pinned by
`tests/test_frontend_semantics.py`.

- **Durative actions merged their at-start and at-end effects**, and an add
  beat a delete, so `(at start (busy)) (at end (not (busy)))` ended with
  `(busy)` true. The at-end effects now apply after the at-start ones and win a
  conflict, and over-all and at-end conditions are checked in the state after
  the at-start effects rather than before them. `always` is checked in that
  state too; the other trajectory constraints are refused when a durative
  action has at-start effects, because their monitors cannot see it.
- **`sometime-after` could not be met by the final state.** An obligation
  recorded by the last action's monitor could never be discharged, so the
  plan `[t, r]` was reported unsolvable; the goal now accepts the response
  holding at the end.
- **`forall` around a trajectory constraint moved inside it**:
  `(forall (?x) (sometime (lit ?x)))` meant one moment with every `?x` lit at
  once. It is now one constraint per object. A soft constraint under `forall`
  crashed with an `AttributeError`; it is refused by name like the others.
- **A goal preference under `forall` kept its free variable** and could never
  be satisfied; it is now one preference per object. Under `not`, `or`, `imply`
  or `exists` a preference silently made the goal false; it is now an error.
- **Timed initial literals slipped past the PDDL 3 machinery.** The
  `at-most-once` and `sometime-after` monitors ignored the states literals
  fired in, a literal could fire after the preferences had been paid out, and
  a plan could end before a literal that was already due had fired.
- **Object fluents in trajectory constraints** were read as numeric
  comparisons; they are now compiled as everywhere else. Assigning one object
  fluent's value to another took the function's name for an object; it is
  refused.
- **The metric.** A preference the metric does not mention cost 1 to violate
  rather than nothing, a metric without `(total-cost)` still charged for
  actions, `maximize` was silently minimised, and `(total-time)` charged an
  instantaneous action 1. `maximize`, any metric other than `(total-cost)` or
  `(total-time)` plus weighted `(is-violated p)` terms, and a metric pricing a
  preference the goal does not declare are now refused.
- **`ground()` edited the domain it was given**, so a second problem grounded
  against the same parsed domain inherited the first one's synthetic actions
  and lost the domain's own constraints.
- **Misspelt PDDL was accepted.** `:preconditon` gave an action with no
  precondition and `(:goals ...)` a problem with an empty goal; unknown keys
  and sections and a missing `:goal` are now errors that name them. Malformed
  input — `(?x -)`, `(when (q))`, `(:goal)` — raised a bare `IndexError`; it is
  a `PDDLError`, and an unbalanced parenthesis names its line.
- **Objects called `nan`, `inf` or `infinity` were read as numbers.**
- **`(increase (total-cost) (road-length ?a ?b))` was refused** although
  `:action-costs` is listed as native. A cost over static functions is now
  evaluated when grounding, so the IPC `transport` domains load. `:fluents`
  now implies `:object-fluents`, and `:ucpop`, `:domain-axioms` and
  `:safety-constraints` are refused by name rather than as unknown flags.

## [2.4.0]

### Added
- **A Research view in the workbench.** `jupyddl.learn` shipped in 2.3.0 with
  its results only in `.docs/` and a video, which meant the published site said
  nothing about the biggest addition in the release. The view carries the
  measured run: imitation against `hff` and `goalcount`, the per-instance
  spread behind the mean, the logistics loss and why the feature space causes
  it, and the three claims that turned out to be wrong. Its numbers are built
  from `promo/rl-data.json` — the same cache the RL video renders from — so
  the page and the video cannot drift apart, and a test pins them together.

### Fixed
- **Reading the workbench no longer costs a 10 MB download.** `<main>` was
  hidden until Pyodide reported ready, so every page of prose, the requirement
  matrix and the new Research view sat behind a full-screen spinner waiting for
  a runtime none of them use. The shell now renders immediately from the
  committed bundle, the boot notice is a status bar rather than a splash
  screen, and only the controls that actually run a planner stay disabled until
  the interpreter arrives.

### Changed
- **Python 3.10 is the new floor**, and the matrix runs 3.10 through 3.14.
  3.9 reached end of life in October 2025; `requires-python` moved to match
  what is actually tested.
- The `pddl-examples` submodule advances to a corpus that now grows on its own:
  it imports fresh domains from planning.domains daily.
- All URLs follow the organisation's rename to `openplan-labs`.

### Removed
- Mergify. Its auto-merge rule waited on an approving review that a
  single-maintainer project does not produce, and its reviewer-assign rule
  pinged an inactive account; every recent pull request was merged by hand.

## [2.3.0] - 2026-07-31

### Added
- **`jupyddl.learn`: heuristics trained from your own solved plans.** Every
  solved instance is a labelled trajectory — the cost of a plan's suffix from a
  state on it is that state's cost-to-go — so a corpus of plans is supervision
  that already exists. Stdlib-only like the rest of the core; `pip install
  jupyddl[learn]` adds NumPy purely for speed, and the two code paths are pinned
  together by a test.
- **A ranking objective, by default.** Greedy best-first search never reads a
  heuristic value, only the order it imposes, so the loss optimises the order:
  at each state on a plan, the successor the plan took should sort ahead of the
  siblings it passed over. A small regression term is kept to anchor a scale,
  which pure ranking leaves undefined and `wastar` needs. Checkpoints are
  selected on top-1 ranking accuracy rather than validation error.
- **A reinforcement stage that optimises search cost directly**, since "nodes
  expanded" is not a differentiable function of the weights: DAgger for the
  distribution shift, bootstrapping for the instances too hard to label, and
  the cross-entropy method over the weight vector with the planner as a black
  box. On blocksworld this took a held-out set from 366 expansions and 0.90
  coverage to 137 and 1.00 — though see `.docs/rl-for-search.md`: that mean is
  dominated by one hard instance, and the durable claim is the coverage one.
- **`jupyddl learn`**, and `learned:<model.json>` accepted anywhere a heuristic
  name is — `solve`, `benchmark`, the API. `make_heuristic` also passes through
  an already-built heuristic, so callers holding a trained model need not
  round-trip it to disk.
- `.docs/` — research notes: the prior work, the measured results including the
  domain where this loses badly and exactly why, the MDP the RL stage
  corresponds to, and a roadmap.

### Fixed
- **`always` was not enforced across the actions timed initial literals compile
  to.** The invariant went onto the domain's own actions and the goal, on the
  argument that every state is either taken from by an action or is the final
  one — but the firing and waiting actions a timed literal compiles to also
  change facts, and they were exempt. A literal clearing `(safe)` at t=3 and one
  restoring it at t=4 fire back-to-back, so the unsafe state in between was
  never checked and the plan was reported valid. Invariants now go onto every
  action.
- **The browser bundle is byte-reproducible again.** `tools/build_web.py`
  sorted the files inside each directory but let `os.walk` visit the
  subdirectories in filesystem order, so `jupyddl-sources.json` came out
  byte-different on a CI runner and the staleness check failed on a bundle that
  was not stale.
- **`format` no longer strands `pages`.** It reformats `jupyddl/` and commits to
  main; `web/dist` embeds those sources verbatim and `pages` refuses to deploy a
  bundle that has drifted, so the reformat could leave main undeployable. The
  rebuild now rides along in the same commit.
- **The sdist shipped 6.6 MB of marketing.** Hatchling's defaults packed the
  promo video, every screenshot and every documentation chart alongside a 105 KB
  wheel. Now 235 KB, and it still builds, installs and passes its own suite —
  which two parser tests previously broke, hard-failing without the
  `pddl-examples` submodule instead of skipping like the other 156.
- **Mergify's auto-merge rule matched no check.** It required
  `check-success=tests`, which is the *workflow* name — the checks are the
  matrix jobs (`test (ubuntu-latest, 3.12, dev)`), so the rule never fired.

### Changed
- CI runs the suite on Python 3.13 and 3.14 as well as 3.9–3.12, and lints
  `tools/` alongside `jupyddl` and `tests`.
- One matrix entry installs the `viz` extra. `tests/test_viz.py` and
  `tests/test_cli_viz.py` `importorskip` matplotlib, so 26 of their 30 tests —
  the entire charting surface — were silently skipped in CI.
- `build` no longer re-runs the lint the `tests` matrix already ran. It builds
  the sdist and wheel, `twine check`s them, installs the wheel into a clean
  environment and plans with it from outside the repository — packaging
  breakage an editable install cannot show.
- `actions/checkout` v4 → v7, `actions/setup-python` v5 → v7,
  `codecov/codecov-action` v4 → v7 (#146, #150, #145); the `format` job runs on
  Python 3.14 (#149); the Mergify config drops the deprecated
  `delete_head_branch` block (#143).

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

# Contributing to PythonPDDL

Thanks for showing up.

This repository is `openplan-labs/PythonPDDL`; the package it publishes is
called **`jupyddl`**, which is what you import and what you `pip install`.

The [org-wide contributing guide](https://github.com/openplan-labs/.github/blob/main/CONTRIBUTING.md)
covers the PR flow, commit style, and what a useful bug report contains. This
page covers only what is specific to this repository.

## Setting up

The `pddl-examples` submodule supplies the instances the tests and benchmarks
read, so clone with it:

```bash
git clone --recurse-submodules https://github.com/openplan-labs/PythonPDDL
```

Python 3.10 is the floor; CI runs 3.10 through 3.14.

```bash
uv venv
uv pip install -e ".[dev,viz,learn]"
```

## What CI will run

```bash
flake8 jupyddl tests tools --count --statistics
pytest
black .            # CI reformats main automatically, but reviewers read diffs
```

The `learn` extra is optional at runtime and its NumPy path is not exercised by
every CI job. If you touch `jupyddl/learn/`, run the suite with it installed.

## Filing a bug

A parser or planner bug needs its input. Attach the **domain and problem
files**, say which **planner and heuristic** you ran (`astar` alone is
ambiguous; `astar` with `lmcut` is not), and give the exact command.

If you are reporting that something is slow, conditions are part of the report:
machine, instance, and how you measured. A number without them is not a result.

## Changing a guarantee

`optimal`, `complete`, `admissible` and `anytime` are load-bearing words here —
`OPTIMAL_PLANNERS` in [`jupyddl/search/__init__.py`](../jupyddl/search/__init__.py)
is derived from an `optimal` attribute on each planner, and the workbench
publishes it. If a change moves a planner in or out of that set, or narrows the
assumption a guarantee holds under, say so in the PR description.

New planners and heuristics should cite the paper they implement — name and
year — in the docstring. It tells a later reader which variant they are
getting.

## Adding an algorithm

This is what the project is for. jupyddl aims to be the place where a planning
algorithm is implemented readably, compared fairly and *explained*. A new
planner, heuristic, compilation or generator is the most welcome contribution
there is.

1. Open an issue with the **Propose an algorithm** template first, so the
   variant and its guarantees are agreed before you write code.
2. Register it in the matching registry: `jupyddl.search.PLANNERS`,
   `jupyddl.heuristics.HEURISTICS` or `jupyddl.generator.GENERATORS`. The CLI,
   the benchmark harness and the browser workbench all read from these, so
   registering it is all the wiring you need.
3. Test the guarantee, not only that it runs. If you claim optimality, compare
   the plan cost against `dijkstra` on small instances. If you claim
   admissibility, check `h(s0)` against the optimal cost.
4. Keep the core stdlib-only. Anything that needs a third-party package goes
   behind an optional extra, the way `viz` and `learn` do.

## Credit

We intend to publish jupyddl as peer-reviewed research software (see
[`.docs/PUBLISHING.md`](../.docs/PUBLISHING.md)). If you make a substantial
contribution and want to be considered for authorship on the paper, say so in
your pull request.

## Licence

This project is licensed under **Apache 2.0** (see [`LICENSE`](../LICENSE)). By
contributing, you agree that your contributions are licensed under the same
terms.

# Extending jupyddl

jupyddl is meant to be built on. This guide adds a heuristic, two planners
and an instance generator, then checks their guarantees, watches them run and
ships them as an installable plugin. Every Python block on this page is run by
`tests/test_extending.py`, in order, so the guide stays accurate.

## The shape of the framework

```
PDDL text ─ parse ─▶ Domain/Problem ─ compile (PDDL 3) ─▶ ground ─▶ Task
                                                                    │
                       Heuristic(task) ─▶ h(state) ─────────────────┤
                                                                    ▼
                 Planner.search(task, heuristic, observer, budget) ─▶ SearchResult
```

A **`Task`** is all a search algorithm needs. Facts are integers, and a state
is a `frozenset` of them, or a `State(facts, values)` when the domain has
numeric fluents.

| You need | Use |
|---|---|
| The start state | `task.initial_state()` |
| Is this a goal? | `task.goal_reached(state)` |
| The actions that apply here | `task.applicable_operators(state)` |
| The successor | `task.apply(op, state)` (**not** `op.apply`: see the contract below) |
| An action's cost and name | `op.cost`, `op.name` |
| The fact set of any state | `facts_of(state)` from `jupyddl.task` |
| The goal facts | `task.goals` |
| A readable fact or state | `task.fact_name(i)`, `task.state_str(state)` |

There are three registries, which are plain dictionaries. The CLI, the
benchmark harness and the browser workbench all read from them, so registering
an algorithm is all the wiring it needs:

- `jupyddl.search.PLANNERS`
- `jupyddl.heuristics.HEURISTICS`
- `jupyddl.generator.GENERATORS`

## A heuristic

A heuristic is built once per task and then called on states. This one is
small but honest. Every goal the state is missing must be added by *some*
action, so any plan costs at least the cheapest action that adds it. That
holds for each missing goal, and so for the most expensive of them:

```python
import math

from jupyddl.heuristics import HEURISTICS, Heuristic
from jupyddl.task import facts_of


class CheapestAchiever(Heuristic):
    """h = max over unmet goals of the cheapest action adding it. Admissible."""

    name = "achiever"
    admissible = True

    def __init__(self, task):
        super().__init__(task)
        self.cheapest = {}
        for op in task.operators:
            added = set(op.add)
            for effect in op.cond_effects:
                added |= effect.add
            for fact in added:
                self.cheapest[fact] = min(op.cost, self.cheapest.get(fact, math.inf))

    def __call__(self, state):
        unmet = self.task.goals - facts_of(state)
        # A goal no action adds may still be derived by an axiom: it counts as
        # 0 here, not inf. Returning inf would prune a state that is solvable.
        return float(max((self.cheapest.get(g, 0) for g in unmet), default=0))


HEURISTICS["achiever"] = CheapestAchiever
```

Return `math.inf` only for a state you have *proved* is a dead end. The
planners prune on it.

## A planner from a priority function

Every best-first planner here is the same engine, `best_first`, with a
different priority over `(g, h)`. A* uses `g + h`, but among nodes with equal
`f` it expands in insertion order. Breaking those ties towards lower `h`
heads for the goal across a plateau, and A* stays optimal, because the order
only changes *within* one `f` layer:

```python
from jupyddl.search import PLANNERS, Planner, best_first, heuristic_name


class TieBreakingAStar(Planner):
    """A* breaking f-ties towards lower h. Optimal with an admissible h."""

    name = "astar-tb"
    requires_heuristic = True
    optimal = True

    def search(self, task, heuristic=None, observer=None, budget=None):
        if observer is not None:
            observer.on_start(task, self.name, heuristic_name(heuristic))
        return best_first(
            task,
            lambda g, h: (g + h, h),
            heuristic=heuristic,
            observer=observer,
            budget=budget,
        )


PLANNERS["astar-tb"] = TieBreakingAStar
```

## A planner from scratch

Writing the loop yourself means honouring the contract that every planner
here keeps:

1. **Successors come from `task.apply(op, state)`**, never `op.apply(state)`.
   The task re-closes derived predicates after each step. Calling the operator
   directly silently skips that.
2. **Poll the budget once per expansion.** Python cannot interrupt a running
   loop, so a `--time-limit` only works if you ask. `budget.exceeded(stats)`
   sets `stats.truncated` itself. A truncated run must never be reported as
   unsolvable.
3. **Touch the observer only behind `if observer is not None`.** Search without
   an observer must cost exactly what it did before observers existed. With
   one, it must return the same plan and statistics.
4. **Fill in `SearchStats`** (`expanded`, `generated`, `evaluated`, `runtime`),
   so the benchmark harness can compare you with everything else.
5. **Claim only what holds.** Set `optimal` and `requires_heuristic` truthfully,
   and cite the paper you implement in the docstring.

Random walks make a good baseline. Monte-Carlo random walks (Nakhost and
Müller, IJCAI 2009) are a serious planner when a heuristic guides them; this
is the unguided core:

```python
import random
import time

from jupyddl.search import SearchResult, SearchStats


class RandomWalk(Planner):
    """Restarting random walks (after Nakhost & Müller, 2009). Not complete."""

    name = "walk"

    def __init__(self, walks=500, length=60, seed=0):
        self.walks, self.length, self.seed = walks, length, seed

    def search(self, task, heuristic=None, observer=None, budget=None):
        stats = SearchStats()
        started = time.perf_counter()
        rng = random.Random(self.seed)  # same seed, same plan
        if observer is not None:
            observer.on_start(task, self.name, heuristic_name(heuristic))
        found = None
        for _ in range(self.walks):
            found = self._walk(task, rng, stats, observer, budget)
            if found is not None or stats.truncated:
                break
        stats.runtime = time.perf_counter() - started
        if found is None:
            result = SearchResult(False, None, None, stats)
        else:
            plan, cost, state = found
            result = SearchResult(True, plan, cost, stats)
            if observer is not None:
                observer.on_goal(state, cost, len(plan), stats)
        if observer is not None:
            observer.on_finish(result)
        return result

    def _walk(self, task, rng, stats, observer, budget):
        state, plan, cost = task.initial_state(), [], 0
        for _ in range(self.length):
            if task.goal_reached(state):
                return plan, cost, state
            if budget is not None and budget.exceeded(stats):
                return None  # stats.truncated is now set
            stats.expanded += 1
            if observer is not None:
                observer.on_expand(state, g=cost, f=cost, depth=len(plan), stats=stats)
            ops = list(task.applicable_operators(state))
            if not ops:
                return None  # stuck: the next walk starts over
            op = rng.choice(ops)
            state = task.apply(op, state)
            stats.generated += 1
            plan.append(op)
            cost += op.cost
        return (plan, cost, state) if task.goal_reached(state) else None


PLANNERS["walk"] = RandomWalk
```

## An instance generator

A generator is a function `(size, seed, **kwargs) -> (domain_text,
problem_text)`. It must be **reproducible**: draw every random choice from a
`random.Random(seed)`, never from the global generator. The first line of its
docstring is what `jupyddl generate` and the workbench show.

```python
from jupyddl.generator import GENERATORS

CORRIDOR = """(define (domain corridor)
  (:requirements :strips :typing)
  (:types cell)
  (:predicates (at ?c - cell) (adjacent ?a ?b - cell))
  (:action move
    :parameters (?from ?to - cell)
    :precondition (and (at ?from) (adjacent ?from ?to))
    :effect (and (at ?to) (not (at ?from)))))"""


def generate_corridor(size=6, seed=0):
    """A robot walks a corridor of `size` cells to a random goal cell."""
    rng = random.Random(seed)
    cells = [f"c{i}" for i in range(size)]
    start, goal = rng.sample(cells, 2)
    links = " ".join(
        f"(adjacent {a} {b}) (adjacent {b} {a})" for a, b in zip(cells, cells[1:])
    )
    problem = f"""(define (problem corridor-{size}-{seed})
  (:domain corridor)
  (:objects {' '.join(cells)} - cell)
  (:init (at {start}) {links})
  (:goal (at {goal})))"""
    return CORRIDOR, problem


GENERATORS["corridor"] = generate_corridor
```

## Run it, and check what you claimed

Everything is registered, so the normal entry points can use it by name:

```python
from jupyddl import ground, parse, solve_task, validate_plan
from jupyddl.generator import generate


def make_task(kind, size, seed):
    domain, problem = generate(kind, size=size, seed=seed)
    return ground(parse(domain), parse(problem))


task = make_task("corridor", 10, 6)
result = solve_task(task, "astar-tb", "achiever")
print(result.cost, result.plan_names())
```

A guarantee is a claim, so test it the way a reviewer would. Check the
optimal planner's cost against `dijkstra`, which is optimal with no heuristic
at all. Check the admissible heuristic's estimate at the start against that
cost. And check that every plan the incomplete planner returns is valid:

```python
for seed in range(5):
    task = make_task("blocksworld", 4, seed)
    optimum = solve_task(task, "dijkstra").cost
    assert solve_task(task, "astar-tb", "achiever").cost == optimum
    assert CheapestAchiever(task)(task.initial_state()) <= optimum
    walk = solve_task(task, "walk")
    assert not walk.solved or validate_plan(task, walk.plan)

# A budget stops it, and says so rather than claiming there is no plan.
stopped = solve_task(make_task("blocksworld", 6, 0), "walk", max_expansions=50)
assert stopped.truncated and not stopped.solved
```

## Watch it, and explain it

The point of putting an algorithm here is that you can *show* it.
`trace_search` records every expansion and works with any planner that
follows the observer contract, including yours:

```python
from jupyddl import trace_search

result, trace = trace_search(make_task("blocksworld", 5, 1), "astar-tb", "achiever")
print(trace.summary()["expanded"], "expansions")
```

With the `viz` extra installed, the same trace turns into charts and a video.
See [Recording and plotting a search](../README.md#recording-and-plotting-a-search)
in the README. From the terminal, `jupyddl solve ... --live` shows the search
as it runs.

## Ship it as a plugin

Assigning into a registry only affects the process that runs the assignment.
To make your algorithms available to everyone who installs your package,
declare **entry points** in its `pyproject.toml`:

```toml
[project.entry-points."jupyddl.planners"]
astar-tb = "my_package.search:TieBreakingAStar"
walk = "my_package.search:RandomWalk"

[project.entry-points."jupyddl.heuristics"]
achiever = "my_package.heuristics:CheapestAchiever"

[project.entry-points."jupyddl.generators"]
corridor = "my_package.generators:generate_corridor"
```

After `pip install my-package`, they work everywhere, without anyone
importing your code:

```bash
jupyddl plugins                                   # what was found, and from where
jupyddl generate corridor --size 12 -o instances/
jupyddl solve instances/corridor-12-0/domain.pddl instances/corridor-12-0/problem.pddl \
    -s astar-tb -H achiever
jupyddl benchmark instances/ --planners astar,astar-tb --heuristic achiever
```

Discovery is lazy and defensive (see `jupyddl/plugins.py`):

- A name that already exists always wins, so a plugin cannot replace a
  built-in.
- A plugin that fails to import is skipped with a warning. It cannot break
  jupyddl.
- `JUPYDDL_NO_PLUGINS=1` switches discovery off, for a benchmark that must not
  depend on what else is installed.

## Contributing it upstream instead

An algorithm that teaches something the existing ones do not belongs in
jupyddl itself. Open a **Propose an algorithm** issue first, then see
[CONTRIBUTING.md](../.github/CONTRIBUTING.md#adding-an-algorithm).

# What to do next

Ordered by expected value. Each entry says what the experiment is, so that a
negative result is as informative as a positive one.

---

## 1. Relational features — fix logistics

**Why first.** The logistics failure is not a tuning problem, it is a
representational one, and it is measured rather than suspected: the domain has
two predicates, so the feature vector is 8 numbers, and two states with a
package at its destination and across the map are *identical* under it. Top-1
ranking accuracy 0.656. No amount of training fixes an input that has thrown the
problem away.

**What to build.** Weisfeiler-Leman features over the grounded problem graph,
following GOOSE (Chen, Trevizan and Thiébaux, ICAPS 2024). The striking finding
there is that classical ML over WL features beat the deep relational models it
was compared against — which, if it holds, is exactly the right shape for this
library: colour refinement is a few passes of hashing over an adjacency
structure, cheap enough to run per state in Python.

Sketch: build a graph with a node per object and per ground atom, edges from
atoms to their arguments, node labels from predicate symbol and goal membership.
Run `k` rounds of colour refinement. The feature vector is the histogram of
colours. Fixed length via a hash into `d` buckets, so it stays size-invariant.

**The experiment.** Same three domains, same protocol. Success is logistics
top-1 above 0.85 and expansions within 2× of `hff`, *without* regressing
blocksworld or gripper. Watch the per-state cost: if it exceeds ~3× the current
features, the wall-clock advantage that makes this whole line worthwhile is gone.

**Risk.** Colour refinement per state may simply be too slow in CPython. Measure
before building the rest.

---

## 2. Reward shaping with `h_ff`

**Why.** The credit-assignment problem in [rl-for-search.md](rl-for-search.md)
is the reason we use a derivative-free method rather than a policy gradient.
Gehring et al. (ICAPS 2022) attack it directly by using a classical heuristic as
a dense reward generator. It is the most promising unexplored direction and it
fits the existing structure with no new machinery.

**What to build.** A potential-based shaping term `F(s, s′) = γ·Φ(s′) − Φ(s)`
with `Φ = −h_ff`. Potential-based shaping is policy-invariant (Ng, Harada and
Russell, ICML 1999), so it changes the learning dynamics without changing what
the optimal policy is — which is exactly the property you want and exactly what
naive shaping gets wrong.

**The experiment.** Compare CEM-with-shaping against plain CEM at equal wall
clock, not equal iterations. Shaping that needs an `h_ff` evaluation per state
is not free.

---

## 3. A proper evaluation protocol

**Why.** Everything in these notes is single-seed on generated instances. It
supports "this works, and here is exactly when it does not". It does not support
a performance claim, and pretending otherwise is how learned-planning results
get published and then fail to replicate.

**What to build.**

- IPC benchmark domains, not just the generators. The generators produce
  instances with a regularity real domains do not have, and a feature space
  keyed on predicate counts is exactly the sort of thing that exploits it.
- A fixed train/test split *per domain*, declared before running anything.
- Five seeds, report median and interquartile range.
- Coverage under a declared budget, plus expansions and wall-clock, plus plan
  cost, on every table. Reporting expansions alone is the standard way to
  publish a learned heuristic that is slower than the one it replaced.
- A per-node cost measurement — microseconds per heuristic evaluation — as a
  first-class number.

**The experiment.** This *is* the experiment. Expect the picture to get worse:
generated instances flatter the approach.

---

## 4. Learning from failure, not just from plans

**Why.** The corpus currently contains only states on successful plans. The
information in "the search spent 4000 expansions in this region and found
nothing" is thrown away, and it is arguably more valuable — it identifies the
plateaux and dead ends where a heuristic is actually costing you.

**What to build.** Record, per expanded state, whether it appeared on the final
plan. Train an auxiliary head to predict "is this state on a solution path",
and add a penalty term for confident low estimates on states that turned out to
be dead ends. Related in spirit to Ferber et al.'s work on progress states in
GBFS.

**The experiment.** Does it reduce plateau escape time? Measure expansions
between successive improvements in best-`h`, not just total expansions — a
heuristic can have identical total cost and a very different plateau profile.

---

## 5. Portfolio / per-instance selection

**Why.** The three-domain result is not "learned beats `hff`", it is "learned
beats `hff` on two of three, and loses badly on the third". That is a selection
problem, and selection problems are usually easier than the underlying one.

**What to build.** A cheap classifier over instance-level features (predicate
count, object count, goal count, `h_ff` of the initial state, branching factor
at the root) predicting which heuristic to use. `jupyddl` already has the
benchmark harness that generates the training data for this.

**The experiment.** Does the selector beat always-`hff` and always-learned on a
held-out set of domains? Note the honest baseline: a selector that always picks
`hff` is the thing to beat, and on three domains it wins one-third of the time
by construction.

---

## 6. Make the browser workbench train

**Why.** `jupyddl/learn` is stdlib-only, so it already runs under Pyodide. A
"watch a heuristic learn, then watch it search" view would make the whole line
legible in a way a table of numbers does not.

**What to build.** A fifth view in `web/`: pick a domain, generate a ladder,
watch the corpus fill, watch the loss curve and the top-1 accuracy, then race
the trained heuristic against `hff` on a larger instance with the existing
wavefront visualisation.

**Cost.** `tools/build_web.py` currently skips `jupyddl/learn` to keep the
bundle small. Un-skipping it is one line. Training in WebAssembly without NumPy
will be slow — budget for a demo-scale ladder, not a real one.

---

## Things deliberately not on this list

- **Deep relational architectures (ASNets, HGNs) as implemented in the
  literature.** They need a tensor library, which breaks the zero-dependency
  core, and their per-state cost is hard to justify in CPython. Item 1 is the
  version of this idea that fits here.
- **Learning the search algorithm rather than the heuristic.** Interesting, much
  larger, and it would not compose with the fourteen planners already in the
  registry.
- **LLM-based planning.** A different research programme, not this one.

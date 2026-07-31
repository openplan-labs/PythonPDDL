# Learning a heuristic from solved plans

## The idea, and why it is not new

A heuristic `h(s)` estimates the cost from `s` to a goal. A plan
`s0 →a1 s1 →a2 … →an sn` hands you `n+1` labelled examples for free: the cost of
the suffix from `si` bounds `h*(si)` above, and equals it when the plan is
optimal. Every planning run you have ever done was quietly generating training
data.

The idea goes back at least to Ernandes and Gori (2004), who trained neural
networks as "sub-symbolic" heuristics for search, and to Yoon, Fern and Givan
(2008), who learned control knowledge for forward search from solved instances.
The modern line splits along one axis: **what does the network see?**

- **Hand-designed features.** Cheap, fixed-length, transfers across instance
  sizes by construction. Loses relational structure.
- **Learned relational representations.** ASNets (Toyer et al., AAAI 2018) build
  a network whose structure mirrors the domain's action schemas. STRIPS-HGN
  (Shen, Trevizan and Thiébaux, ICAPS 2020) runs a hypergraph network over the
  delete-relaxation hypergraph, learning a domain-independent heuristic from
  scratch. GOOSE (Chen, Trevizan and Thiébaux, ICAPS 2024) makes the striking
  point that classical ML over Weisfeiler-Leman graph features beats the deep
  models it was benchmarked against, at a fraction of the cost.

`jupyddl` sits deliberately at the cheap end, for a reason worth stating
plainly: **a heuristic that is slower to evaluate than `hff` has to be much
better informed to win on time**, and in a pure-Python planner it will not be.
`lmcut` on a 14-block instance expands 203 nodes in 11.3 seconds; `hff` expands
577 in 1.6; the learned heuristic here expands 819 in 0.32. The *worst*
heuristic of the three by expansions wins by a factor of 5 on the clock. Any
design that costs a graph-network forward pass per generated state has to
overcome that, and nothing about running it in CPython helps.

## What we built

### Features (`jupyddl/learn/features.py`)

Everything is keyed on the **predicate symbol**, never the ground atom, and
normalised by how many atoms of that symbol the task contains. For each symbol
`p` in a fixed vocabulary:

- the fraction of `p`-atoms true in the state,
- the fraction of `p`-goals still unsatisfied,

plus four global features (unsatisfied-goal fraction, its log, state size as a
fraction of all facts, and whether any goal is satisfied at all).

That is `2P + 4` numbers, independent of object count. A model trained on four
blocks can be *evaluated* on forty — which a one-hot over ground atoms makes
impossible, since the vector length and the meaning of every slot change with
the instance.

Cost is `O(|s| + |goals|)`, no successor generation. That matters: it is
evaluated once per generated state.

### The network (`jupyddl/learn/model.py`)

A small ReLU MLP, default `(features → 32 → 16 → 1)`, roughly 1000 parameters.
Output through softplus so the heuristic is never negative — a negative
cost-to-go does not merely mislead a planner, it breaks the assumptions of
every planner consuming it.

Forward and backward passes are written twice: a plain-Python reference and a
NumPy batched version. The core of `jupyddl` has no runtime dependencies and
this keeps that promise; NumPy is a speed option worth one to two orders of
magnitude. `tests/test_learn.py` asserts the two produce identical gradients,
and checks both against finite differences.

### The objective (`jupyddl/learn/train.py`)

This is the part worth arguing about.

The obvious objective is regression: fit `h(s) ≈ h*(s)`, minimise MSE. But
**greedy best-first search never reads a heuristic value.** It reads the *order*
the heuristic imposes on the open list. A model that is uniformly 30 too high
scores terribly on RMSE and guides perfectly. A model with excellent RMSE that
inverts two siblings sends the search down the wrong subtree.

So the default objective here is a **ranking loss**, following Chrestien et al.
(NeurIPS 2023), whose title says it outright: *Optimize Planning Heuristics to
Rank, not to Estimate, Cost-to-Goal*. The idea is older than the deep-learning
framing — Xu, Fern and Yoon learned linear ranking functions for beam search a
decade and a half earlier (IJCAI 2007, JMLR 2009).

Concretely: at each state on a training plan, take the successor the plan
actually took plus the siblings it passed over, and apply a softmax
cross-entropy over `−(c(s,s′) + h(s′))` with the plan's successor as the target.
The step cost has to be in there — GBFS compares `h` values, but the *correct*
comparison is `c + h*`, and with non-uniform action costs those differ.

The default is not *pure* ranking. A ranking loss is invariant to any monotone
rescaling of the output, so it pins down no scale at all. That is fine for GBFS
and useless for weighted A*, where the weight multiplies a quantity that now
means nothing. `rank_weight=0.8` splits the objective; 0.8 is a reasonable
default, not a tuned optimum.

The metric we select checkpoints on is **top-1 ranking accuracy**, not
validation MAE, for the same reason.

### Not admissible, and it does not pretend to be

Nothing in the training objective bounds the prediction from above. One
over-estimate costs A* its optimality guarantee. `LearnedHeuristic.admissible`
is `False` and the docs say to pair it with `gbfs`, or `wastar` if you want a
bounded-suboptimality knob. The suite asserts that a plan found with a learned
heuristic still validates — the heuristic may be wrong, the planner may not
become unsound.

## Results

Train on a small ladder, test on instances two to three times larger with
different seeds. Planner is `gbfs` throughout, budget 30 000 expansions /
30 s. Coverage is the fraction solved; expansions and seconds are means over
solved instances.

### blocksworld — trained on 3–6 blocks, tested on 9–13

| heuristic | coverage | expansions | seconds | plan cost |
|---|---|---|---|---|
| **learned** (imitation only) | 0.90 | 366 | 0.141 | 49.1 |
| **learned** (+ search-cost tuning) | **1.00** | **137** | **0.038** | **48.2** |
| `hff` | 1.00 | 518 | 0.561 | 51.8 |
| `goalcount` | 1.00 | 2483 | 0.169 | 51.0 |
| `blind` | 0.00 | — | — | — |

Against `hff`: **3.8× fewer expansions, 15× faster, and slightly cheaper plans.**
Corpus construction took 0.5 s, training 0.1 s, search-cost tuning about 50 s.
Held-out MAE 0.92, top-1 accuracy 0.923. The model has 1025 parameters over 14
features.

The tuned row is what this prints, end to end, in about a minute:

```bash
jupyddl learn blocksworld --sizes 3-6 --seeds-per-size 3 \
    --cem 10 --cem-sizes 9-12 --evaluate 9-13 -o bw.heur.json
```

Three disjoint instance families are involved and the separation matters:
training uses seed 0, CEM tunes on seed 1000 and selects on seed 2000, and the
evaluation above is seed 7777. Nothing in the table was seen by any stage.

Imitation alone left one test instance unsolved. The reinforcement stage fixed
it — see [rl-for-search.md](rl-for-search.md), including the version of this
run that made transfer five times *worse* before the validation split went in.

### gripper — trained on 2–5 balls, tested on 8–12

| heuristic | coverage | expansions | seconds | plan cost |
|---|---|---|---|---|
| **learned** | 1.00 | **66** | **0.014** | **29.4** |
| `hff` | 1.00 | 135 | 0.078 | 37.0 |
| `goalcount` | 1.00 | 142 | 0.006 | 39.0 |
| `blind` | 0.40 | 20198 | 0.480 | 25.0 |

The cleanest win: half the expansions of `hff`, 5.6× faster, and *better plans*.
Top-1 accuracy 1.000 — gripper's structure is almost entirely captured by "how
many balls are still in the wrong room", which is exactly what these features
encode.

Note `blind`'s plan cost of 25.0, the lowest in the table. It solved only 40% of
the instances and its search is effectively breadth-first on those, so it finds
near-optimal plans for the few it manages. Comparing mean cost across
heuristics with different coverage is misleading, which is why coverage is the
first column.

### logistics — trained on size 2–4, tested on 5–7

| heuristic | coverage | expansions | seconds | plan cost |
|---|---|---|---|---|
| **learned** | 1.00 | 204 | 0.064 | 79.0 |
| `hff` | 1.00 | **35** | **0.029** | **45.7** |
| `goalcount` | 1.00 | 1374 | 0.080 | 102.0 |

**A clear loss.** `hff` expands 6× fewer nodes and finds plans 42% cheaper. The
learned heuristic beats `goalcount` and nothing else.

## Why logistics fails, precisely

This is the most useful result in the set, because the mechanism is exact
rather than hand-wavy.

The generated logistics domain has **two predicates**: `at` and `in`. The
feature vector is therefore `2×2 + 4 = 8` numbers. And those eight numbers
cannot distinguish *which* package is at *which* location — only *how many*
packages are somewhere, and how many goals remain.

Two states where a package sits at its destination and where it sits across the
map produce **identical feature vectors** if the counts match. The heuristic is
being asked to estimate distance-to-goal from a description that has thrown away
the entire structure of the problem. Top-1 ranking accuracy is 0.656, barely
above chance for the branching factor involved, and the model is doing about as
well as anything could on that input.

Blocksworld and gripper survive because their difficulty is closer to
count-shaped: "how many blocks are on the wrong thing", "how many balls are in
the wrong room". Logistics' difficulty is *relational* — it is about the
topology of who needs to go where — and counting is blind to topology.

This is the argument for relational architectures stated as a measurement rather
than an intuition. It is precisely the gap STRIPS-HGN and GOOSE exist to close,
and it says clearly what the next step here has to be: features that survive a
permutation of the objects but not a permutation of the *relations between*
them. See [roadmap.md](roadmap.md).

## Honest limitations

- **Single seed, generated instances, three domains.** These numbers support
  "this works and here is when it does not". They do not support a performance
  claim. A publishable evaluation needs IPC benchmark domains, multiple seeds
  with confidence intervals, and a per-domain train/test protocol fixed in
  advance.
- **Coverage is measured under a budget.** A heuristic that solves 90% within
  30 000 expansions may solve 100% at 300 000. Coverage numbers here are "within
  this budget", not "ever".
- **Plan cost is not controlled.** GBFS is not optimal with any of these
  heuristics, so cost comparisons across rows compare two different suboptimal
  behaviours, not two approximations of the same thing.
- **The corpus is small.** 118–166 samples per domain. The model has ~1000
  parameters. That it generalises at all is mostly a statement about how much
  structure the features already impose.
- **Training targets on logistics were expensive to get.** Optimal solving of
  the size 2–4 ladder took 13.9 s versus 0.5 s for blocksworld, and this
  worsens sharply with size. Bootstrapping (see the RL note) exists for exactly
  this.

## References

- Ernandes, M. and Gori, M. *Likely-admissible and sub-symbolic heuristics.* ECAI 2004.
- Yoon, S., Fern, A. and Givan, R. *Learning control knowledge for forward search planning.* JMLR 2008.
- Xu, Y., Fern, A. and Yoon, S. *Discriminative learning of beam-search heuristics for planning.* IJCAI 2007.
- Xu, Y., Fern, A. and Yoon, S. *Learning linear ranking functions for beam search with application to planning.* JMLR 2009.
- Arfaee, S. J., Zilles, S. and Holte, R. C. *Learning heuristic functions for large state spaces.* Artificial Intelligence 2011.
- Toyer, S., Trevizan, F., Thiébaux, S. and Xie, L. *Action schema networks: Generalised policies with deep learning.* AAAI 2018.
- Shen, W., Trevizan, F. and Thiébaux, S. *Learning domain-independent planning heuristics with hypergraph networks.* ICAPS 2020.
- Ferber, P., Helmert, M. and Hoffmann, J. *Neural network heuristics for classical planning: A study of hyperparameter space.* ECAI 2020.
- Chrestien, L., Edelkamp, S., Komenda, A. and Pevný, T. *Optimize planning heuristics to rank, not to estimate, cost-to-goal.* NeurIPS 2023.
- Chen, D. Z., Trevizan, F. and Thiébaux, S. *Return to tradition: Learning reliable heuristics with classical machine learning.* ICAPS 2024.

These are cited from working knowledge of the literature. Verify each against
the published record before any of this is submitted anywhere.

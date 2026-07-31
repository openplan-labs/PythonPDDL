# When learning a heuristic becomes reinforcement learning

## The moment it changes

Imitation asks *what does `h*` look like?* and fits a network to labels.

The question we actually care about is *which heuristic makes search expand the
fewest nodes?* — and the answer is not the same. Imitation optimises a proxy,
and the proxy is wrong in two specific ways:

1. **It is trained on the wrong distribution.** The corpus contains states that
   lie on optimal plans. At search time the planner asks about states that do
   not: the dead ends, the detours, the plateaux a mediocre heuristic wanders
   into. This is covariate shift, and it is the standard failure mode of
   behavioural cloning.
2. **It optimises the wrong quantity.** Fitting `h*` well is neither necessary
   (a monotone rescaling changes nothing for GBFS) nor sufficient (a small error
   at a critical branch point costs a whole subtree).

The moment you stop fitting labels and start optimising *the number of nodes
expanded*, this is reinforcement learning. The heuristic is a policy; the
planner is the environment; the reward is the search you did not have to do.

## The MDP

Formally, for greedy best-first search:

| | |
|---|---|
| **State** | the search state — the open list, the closed set, everything the planner knows |
| **Action** | which node to expand next |
| **Policy** | induced by `h`: expand `argmin` over the open list |
| **Transition** | pop the node, generate successors, push them |
| **Reward** | `−1` per expansion; terminal on reaching a goal |
| **Return** | `−(nodes expanded)` |

Maximising return is exactly minimising search effort. The parameters of `h`
parameterise the policy, so this is policy search over a
non-differentiable objective.

Note what the reward is *not*: it is not plan quality. A GBFS guided to a goal
in twelve expansions along a wasteful path scores better than one that takes two
hundred to find the optimal plan. If plan quality matters, it has to be in the
reward — `−(expansions) − λ·(plan cost)` — and it is not, by default, here.

## Why the obvious policy gradient is hard

REINFORCE needs a stochastic policy. The natural one is Boltzmann over the open
list: expand node `i` with probability `∝ exp(−h(si)/τ)`. Then

```
∇θ J = E[ Σt ∇θ log π(at | st) · Gt ]
```

Three things go wrong.

**The episode is thousands of steps long, and every step gets the same credit.**
Expansion number 3 and expansion number 3000 receive identical returns under a
uniform `−1` reward. The variance of the estimator scales with episode length,
and episode length is the quantity being optimised — so the estimator is worst
exactly where the policy is worst.

**The action space is the open list, which grows.** `π` is a softmax over
tens of thousands of items, changing size at every step. Not fatal, but it means
each gradient step touches every node in the frontier.

**Making the policy stochastic makes the planner worse.** Deterministic
`argmin` is the thing we ship. Training a Boltzmann policy and deploying an
`argmin` one is a train/test mismatch on top of everything else, and as `τ → 0`
to close it, the gradient vanishes.

Gehring et al. (ICAPS 2022) address the credit-assignment problem head-on by
using `h_ff` as a *dense reward generator* rather than relying on the sparse
goal signal — reward shaping with a domain-independent heuristic. That is a
good idea and the obvious next thing to try here.

## What we do instead

Three mechanisms, addressing different failures. All are in
`jupyddl/learn/rl.py`.

### 1. DAgger — fix the distribution

Run the current heuristic. Keep the states it *expanded*. Label each by solving
from it. Retrain on the union. Repeat. (Ross, Gordon and Bagnell, AISTATS 2011.)

Labelling a state means finding a plan from it, which needs the planner to start
somewhere other than `task.initial_state()`. Rather than thread a start state
through fourteen planner signatures, `task_from_state` re-roots the task —
operators, goals, axioms and the numeric layer are untouched, so every planner
and heuristic works on the result unmodified.

Labels come from a satisficing solver, so they are upper bounds; the samples are
tagged `optimal=False` and down-weighted during training.

**Measured, on blocksworld:** one round with 15 states per task moved search
cost on the *training* instances from 89 to 15 expansions — a 6× improvement —
and moved a held-out set from 89 to **164**, i.e. it got 1.8× worse.

That is a real result and it is worth stating plainly: **DAgger overfitted.**
With a 12-instance training ladder and 1066 aggregated samples from satisficing
labels, the model specialised to the states those twelve searches happened to
visit. The mechanism is sound and the setting was too small for it. It is off by
default.

### 2. Bootstrapping — fix the data scarcity

You cannot label a 20-block instance you cannot solve. But a heuristic trained
on 6 blocks may just crack 8, whose plans then teach it 10. Each round attempts
the unsolved instances with the current heuristic, adds what it managed, and
retrains. (Arfaee, Zilles and Holte, AIJ 2011.)

This is a policy-improvement loop where the policy is "which instances can I
solve at all", and it is the honest answer to logistics taking 13.9 s to
generate a corpus at size 2–4.

### 3. Direct search-cost optimisation — attack the objective itself

Perturb the weight vector, score each candidate by *actually running the
planner*, keep the best fraction, refit the sampling distribution. The
cross-entropy method, with the planner as a black box.

Evolutionary strategies are a legitimate alternative to policy gradients when
rollouts are cheap and the parameter vector is modest (Salimans et al., 2017),
which describes this exactly: ~1000 parameters, and a rollout is one GBFS run
on a small instance.

**Measured, on blocksworld** — 10 iterations, population 12, σ=0.15, ~50 s:

| | tuning set (sizes 9–12) | held-out benchmark (sizes 9–13, third seed family) |
|---|---|---|
| after imitation | 152 | 366 expansions, 0.90 coverage |
| after CEM | **100** | **137 expansions, 1.00 coverage** |

CEM fixed the coverage failure imitation left behind, then shaved expansions:
2.7× fewer nodes on instances no stage of training ever saw.

### The failure that produced the validation split

The first version of this selected the incumbent by its score on the tuning
set. It looked excellent:

```
cem iter 10: best 108, incumbent 108 (coverage 1.00)
```

and then scored **1734 expansions** on the held-out benchmark — nearly five
times *worse* than the 366 the imitated heuristic managed before tuning. A
thousand parameters had been fitted to eight instances, and had fitted them.

The fix is the ordinary one and it is worth stating because the derivative-free
framing makes it easy to forget that this is still machine learning: candidates
are *proposed* by their tuning score, since that is what the sampling
distribution refits on, but the incumbent — the thing actually returned — only
moves when a **second, disjoint instance family** improves too. With that in
place the same command produces 137.

`optimise_search_cost` reports both numbers every iteration, so a run that is
fitting its tuning set while losing validation is visible rather than silent:

```
cem iter 7: tuning 101, validation 73, incumbent 73 (coverage 1.00)
```

Passing no `validation_tasks` restores the old behaviour, and with it the old
failure mode. `learn_heuristic` always passes one.

## The three things that decide whether this works

### Start from the imitation solution

Search cost is a *step function* of the weights over most of the space: every
candidate that solves nothing scores identically at `failure_penalty × budget`.
A randomly initialised policy sits in that flat region and no method — CEM,
REINFORCE, anything — gets a signal out of it. Imitation is what puts the
optimiser somewhere the objective can distinguish.

### Tune on instances that have headroom

This one cost us an afternoon and is the most transferable lesson here.

The first CEM run tuned on the *training ladder* and moved the score from 12.83
to 12.75 — noise. The reason: after imitation the heuristic already expands
roughly as many nodes as the plan is long on those instances. There is nothing
left to win, so every perturbation scores the same, and the objective is flat
for the second time, for a different reason.

Re-running on sizes 9–12, where the imitated heuristic expanded 1605 nodes on
average, moved it to 64 — a 25× improvement — and cut a held-out set from 336 to
122.

**Optimise where the search is still bad.** `learn_heuristic` therefore defaults
`cem_sizes` to a rung above the training ladder rather than reusing it.

### Select on instances you are not fitting

The section above. It is the difference between 137 and 1734 on the same
command, and it is the single most consequential line of code in this module.

## Design details worth knowing

- **The failure penalty is `2 × budget`, not `1 ×`.** At exactly `1 ×` the
  optimiser is indifferent between solving an instance at the budget limit and
  not solving it. Greater than 1 means solving slowly always beats giving up.
- **Scores are comparable only at a fixed budget**, since the penalty is a
  multiple of it. Every optimiser here holds the budget fixed.
- **The incumbent is never displaced by a worse validation score.** The
  returned bundle cannot be worse than the one passed in, on the instances it
  was *selected* on. That is a real guarantee once those instances are disjoint
  from the ones being fitted, and was nearly worthless when they were not.
- **The mean of the elites is re-evaluated each iteration**, not assumed to be
  good. CEM's distribution mean is not one of the sampled candidates and can be
  worse than all of them.

## What we have not tried

- **Reward shaping with `h_ff`** as a dense signal (Gehring et al. 2022). The
  most promising unexplored direction, and it fits the existing structure.
- **Policy-guided search with guarantees** (Orseau and Lelis, AAAI 2021), which
  puts a bound on the search effort of a learned policy rather than hoping.
- **Optimising for plan quality as well as effort.** Currently unmodelled.
- **REINFORCE with a learned baseline** over a Boltzmann-softened open list,
  which is the textbook approach and which we skipped for the variance reasons
  above rather than because it is wrong.

## References

- Ross, S., Gordon, G. and Bagnell, D. *A reduction of imitation learning and structured prediction to no-regret online learning.* AISTATS 2011.
- Arfaee, S. J., Zilles, S. and Holte, R. C. *Learning heuristic functions for large state spaces.* Artificial Intelligence 2011.
- Salimans, T., Ho, J., Chen, X., Sidor, S. and Sutskever, I. *Evolution strategies as a scalable alternative to reinforcement learning.* arXiv 2017.
- Orseau, L. and Lelis, L. *Policy-guided heuristic search with guarantees.* AAAI 2021.
- Gehring, C., Asai, M., Chitnis, R., Silver, T., Kaelbling, L. P., Sohrabi, S. and Katz, M. *Reinforcement learning for classical planning: Viewing heuristics as dense reward generators.* ICAPS 2022.
- Rubinstein, R. Y. *Optimization of computer simulation models with rare events.* European Journal of Operational Research 1997. (The cross-entropy method.)

Cited from working knowledge; verify before publication.

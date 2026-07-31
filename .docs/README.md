# Research notes

Working notes for the learning-for-planning line in `jupyddl`. These are
research documents, not user documentation — the user-facing description of
`jupyddl learn` is in the README and the module docstrings.

| Note | What it covers |
|---|---|
| [learned-heuristics.md](learned-heuristics.md) | The imitation stage: prior work, the design we chose, measured results on three domains, and an analysis of the one domain where it loses badly |
| [rl-for-search.md](rl-for-search.md) | The reinforcement stage: the MDP that "minimise expansions" corresponds to, why the obvious policy gradient is hard here, and what we do instead |
| [roadmap.md](roadmap.md) | What to build next, ordered by expected value, with the experiment that would settle each |

## The one-paragraph version

A planner's heuristic is a learned function waiting to happen: every solved
instance is a labelled trajectory, since the cost of a plan's suffix from any
state on it is that state's cost-to-go. Fitting a network to those labels is
*imitation*, and it works — on blocksworld, training on 3–6 block instances
produces a heuristic that beats `hff` on 9–13 block instances by 4× in
expansions and 15× in wall-clock. But imitation optimises a proxy. What we
actually want is the heuristic that makes search expand the fewest nodes, and
that quantity is not a differentiable function of the weights. Optimising it
directly — treating the planner as a black box and the weights as a policy —
is where this becomes reinforcement learning, and on blocksworld it is worth
another 2.7× on top of imitation while fixing a coverage failure that
imitation alone left behind — it solved a held-out instance imitation could not
solve at all.

It also produced the most useful negative result here, which is about us rather
than about planning: an earlier version of these notes credited that improvement
to the validation split. Re-running with one variable at a time showed the
perturbation scale was doing all the work, and that nine of the ten held-out
instances improve either way — the whole headline gap is a single hard instance.
Both corrections are written up in `rl-for-search.md` rather than quietly
edited out.

## The video

[`promo/jupyddl-rl.mp4`](../promo/jupyddl-rl.mp4) is a 97-second tour of this
work. Like the main promo it measures everything at render time — it trains,
reinforces, and re-runs both failure modes — so it cannot drift from these
notes. Rebuild it with::

    python tools/make_learn_promo.py --cache promo/rl-data.json -o promo/jupyddl-rl.mp4

`promo/rl-data.json` is the cached measurement pass; delete it to re-measure.

## Reproducing everything here

```bash
pip install -e ".[dev,learn]"

# the headline blocksworld result, about a minute
jupyddl learn blocksworld --sizes 3-6 --seeds-per-size 3 \
    --cem 10 --cem-sizes 9-12 --evaluate 9-13 -o blocksworld.heur.json

# then use it anywhere a heuristic name is accepted
jupyddl solve domain.pddl problem.pddl -s gbfs -H learned:blocksworld.heur.json
```

Every number in these notes came from a run on this repository at the commit
that introduced them, on one CPU, with `numpy` installed. They are single-seed
measurements on generated instances — enough to support the qualitative claims
made here and **not** enough for a paper. See the roadmap for what a publishable
evaluation would need.

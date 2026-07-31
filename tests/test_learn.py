"""Learned heuristics: features, model, corpus, training and the RL stages.

Everything here runs on tiny budgets. The point is that the machinery is
correct, not that a 20-epoch model on six instances is any good — the quality
claims live in ``.docs/learned-heuristics.md`` where they can be stated with
the numbers that produced them.
"""

from __future__ import annotations

import json
import math
import random

import pytest

from jupyddl.heuristics import make_heuristic
from jupyddl.learn import (
    Corpus,
    FeatureSpace,
    HeuristicBundle,
    LearnedHeuristic,
    MLP,
    TrainConfig,
    build_corpus,
    numpy_available,
    samples_from_plan,
    train,
)
from jupyddl.learn.dataset import task_from_state
from jupyddl.learn.features import predicate_of
from jupyddl.learn.model import Adam
from jupyddl.learn.pipeline import (
    evaluate_transfer,
    solved_corpus,
    summarise_transfer,
    tasks_from_generator,
)
from jupyddl.learn.rl import (
    RLConfig,
    bootstrap,
    dagger,
    optimise_search_cost,
    search_cost,
)
from jupyddl.learn.train import evaluate_ranking


@pytest.fixture(scope="module")
def ladder():
    return tasks_from_generator("blocksworld", range(3, 5), seed=0, seeds_per_size=2)


@pytest.fixture(scope="module")
def space(ladder):
    return FeatureSpace.from_tasks(task for _, task in ladder)


@pytest.fixture(scope="module")
def corpus(ladder, space):
    return solved_corpus(ladder, space=space, time_limit=20.0)


# ==========================================================================
# features
# ==========================================================================
@pytest.mark.parametrize(
    "fact,symbol",
    [
        ("(on b1 b2)", "on"),
        ("(handempty)", "handempty"),
        ("(at truck1 loc2)", "at"),
        ("move(a,b)#2", "move"),
        ("__closed", "__closed"),
    ],
)
def test_predicate_symbols_are_extracted_from_both_spellings(fact, symbol):
    """Facts print Lisp-style and operators functionally; both must parse.

    Getting this wrong does not raise. It gives every ground atom its own
    vocabulary slot, and the feature vector quietly stops being size-invariant.
    """
    assert predicate_of(fact) == symbol


def test_vocabulary_is_predicates_not_atoms(space):
    assert set(space.vocabulary) == {"on", "ontable", "clear", "handempty", "holding"}


def test_feature_vector_is_the_same_length_for_every_instance_size():
    """The whole transfer claim rests on this."""
    small = tasks_from_generator("blocksworld", [3], seed=0)[0][1]
    large = tasks_from_generator("blocksworld", [11], seed=0)[0][1]
    space = FeatureSpace.from_tasks([small, large])
    a = space.bind(small)(small.initial_state())
    b = space.bind(large)(large.initial_state())
    assert len(a) == len(b) == space.size
    assert space.size == 2 * 5 + len(FeatureSpace.GLOBAL_FEATURES)


def test_features_are_bounded_so_scale_does_not_leak_in():
    """Counts are normalised by their per-predicate totals, so 40 blocks and
    4 blocks land in the same range rather than an order of magnitude apart."""
    for size in (3, 8, 14):
        task = tasks_from_generator("blocksworld", [size], seed=1)[0][1]
        space = FeatureSpace.from_task(task)
        vector = space.bind(task)(task.initial_state())
        assert all(0.0 <= v <= 3.0 for v in vector), (size, vector)


def test_a_symbol_outside_the_vocabulary_does_not_shift_the_others():
    """A model must survive a task using a predicate it never saw."""
    task = tasks_from_generator("blocksworld", [3], seed=0)[0][1]
    space = FeatureSpace(["clear", "on", "totally-made-up"])
    vector = space.bind(task)(task.initial_state())
    assert len(vector) == space.size
    assert vector[space.vocabulary.index("totally-made-up")] == 0.0


def test_feature_names_line_up_with_the_vector(space):
    task = tasks_from_generator("blocksworld", [3], seed=0)[0][1]
    assert len(space.names()) == len(space.bind(task)(task.initial_state()))


def test_feature_space_round_trips(space):
    assert FeatureSpace.from_dict(json.loads(json.dumps(space.to_dict()))) == space


# ==========================================================================
# model
# ==========================================================================
def test_gradients_match_finite_differences():
    """The one test that would catch a wrong backward pass."""
    rng = random.Random(3)
    model = MLP([6, 5, 4, 1], seed=2)
    batch = [[rng.gauss(0, 1) for _ in range(6)] for _ in range(9)]
    targets = [abs(rng.gauss(2, 1)) for _ in range(9)]

    def loss_of(m):
        out, _ = m.forward_batch(batch)
        return sum((o - t) ** 2 for o, t in zip(out, targets)) / len(targets)

    outputs, cache = model.forward_batch(batch)
    d_out = [2 * (o - t) / len(targets) for o, t in zip(outputs, targets)]
    grad_w, grad_b = model.backward_batch(cache, d_out)
    analytic = [v for layer in grad_w for row in layer for v in row]
    analytic += [v for layer in grad_b for v in layer]

    flat = model.get_flat()
    eps = 1e-6
    for index in range(len(flat)):
        high = list(flat)
        high[index] += eps
        model.set_flat(high)
        upper = loss_of(model)
        low = list(flat)
        low[index] -= eps
        model.set_flat(low)
        lower = loss_of(model)
        model.set_flat(flat)
        assert abs((upper - lower) / (2 * eps) - analytic[index]) < 1e-6


@pytest.mark.skipif(not numpy_available(), reason="numpy is not installed")
def test_the_numpy_path_agrees_with_the_reference():
    """A fast path that has drifted is worse than no fast path."""
    rng = random.Random(0)
    model = MLP([7, 6, 1], seed=1)
    batch = [[rng.gauss(0, 1) for _ in range(7)] for _ in range(11)]
    d_out = [rng.gauss(0, 1) for _ in range(11)]

    fast_out, fast_cache = model._forward_numpy(batch)
    slow_out, slow_cache = model._forward_python(batch)
    assert fast_out == pytest.approx(slow_out, abs=1e-12)

    fw, fb = model._backward_numpy(fast_cache, d_out)
    sw, sb = model._backward_python(slow_cache, d_out)
    flat = lambda g: [v for layer in g for row in layer for v in row]  # noqa: E731
    assert flat(fw) == pytest.approx(flat(sw), abs=1e-12)
    assert [v for layer in fb for v in layer] == pytest.approx(
        [v for layer in sb for v in layer], abs=1e-12
    )


def test_single_sample_call_matches_the_batch(space):
    rng = random.Random(5)
    model = MLP([4, 3, 1], seed=0)
    batch = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(3)]
    outputs, _ = model.forward_batch(batch)
    assert [model(v) for v in batch] == pytest.approx(outputs, abs=1e-12)


def test_the_output_is_never_negative():
    """A negative cost-to-go breaks every planner that consumes it."""
    model = MLP([3, 3, 1], seed=0)
    model.set_flat([-50.0] * model.num_parameters)
    assert model([1.0, 1.0, 1.0]) >= 0.0


def test_flat_round_trip_preserves_predictions():
    model = MLP([5, 4, 1], seed=7)
    vector = [0.3, -0.2, 0.9, 0.1, 0.0]
    before = model(vector)
    model.set_flat(model.get_flat())
    assert model(vector) == pytest.approx(before)


def test_set_flat_rejects_the_wrong_length():
    model = MLP([3, 2, 1], seed=0)
    with pytest.raises(ValueError):
        model.set_flat([0.0])


def test_adam_reduces_a_loss_it_can_reach():
    rng = random.Random(1)
    model = MLP([3, 4, 1], seed=0)
    optimiser = Adam(model, lr=0.05)
    batch = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(16)]
    targets = [2.0] * 16

    def loss_of():
        out, _ = model.forward_batch(batch)
        return sum((o - t) ** 2 for o, t in zip(out, targets)) / len(targets)

    first = loss_of()
    for _ in range(60):
        outputs, cache = model.forward_batch(batch)
        d_out = [2 * (o - t) / len(targets) for o, t in zip(outputs, targets)]
        optimiser.step(*model.backward_batch(cache, d_out))
    assert loss_of() < first * 0.1


# ==========================================================================
# corpus
# ==========================================================================
def test_plan_suffix_costs_are_the_targets(ladder, space):
    name, task = ladder[0]
    from jupyddl.api import solve_task

    result = solve_task(task, "astar", "lmcut", time_limit=20)
    assert result.solved
    samples, groups = samples_from_plan(task, result.plan, space.bind(task), name)
    assert len(samples) == len(result.plan) + 1
    assert samples[-1].target == 0.0
    assert samples[0].target == pytest.approx(result.cost)
    # Monotonically decreasing towards the goal, by construction.
    assert all(a.target >= b.target for a, b in zip(samples, samples[1:]))
    assert groups, "a plan through a branching state should yield ranking groups"


def test_ranking_groups_hold_the_chosen_successor_first(ladder, space):
    from jupyddl.api import solve_task

    name, task = ladder[0]
    result = solve_task(task, "astar", "lmcut", time_limit=20)
    _, groups = samples_from_plan(task, result.plan, space.bind(task), name)
    for group in groups:
        assert len(group.chosen) == 2
        assert group.others
        assert all(len(other) == 2 for other in group.others)


def test_corpus_splits_by_instance_not_by_sample(corpus):
    """Two states on the same plan are near-duplicates; splitting them across
    train and validation reports memorisation as generalisation."""
    train_set, val_set = corpus.split(validation=0.34, seed=0)
    assert train_set.samples and val_set.samples
    assert not (
        {s.instance for s in train_set.samples} & {s.instance for s in val_set.samples}
    )


def test_corpus_round_trips(corpus, tmp_path):
    path = tmp_path / "corpus.json"
    corpus.save(str(path))
    again = Corpus.load(str(path))
    assert len(again) == len(corpus)
    assert again.space == corpus.space
    assert len(again.groups) == len(corpus.groups)


def test_corpus_refuses_to_merge_mismatched_spaces(corpus):
    other = Corpus(FeatureSpace(["nonsense"]))
    with pytest.raises(ValueError):
        other.extend(corpus)


def test_build_corpus_survives_an_unsolved_instance(space, ladder):
    from jupyddl.search.result import SearchResult

    def refuse(_task):
        return SearchResult(False, None, None)

    empty = build_corpus(ladder, refuse, space=space)
    assert len(empty) == 0


def test_task_from_state_reroots_without_touching_anything_else(ladder):
    from jupyddl.api import solve_task, validate_plan

    _, task = ladder[0]
    result = solve_task(task, "astar", "lmcut", time_limit=20)
    midpoint = task.initial_state()
    for operator in result.plan[: len(result.plan) // 2]:
        midpoint = task.apply(operator, midpoint)

    rerooted = task_from_state(task, midpoint)
    assert rerooted.goals == task.goals
    assert rerooted.operators is task.operators
    tail = solve_task(rerooted, "astar", "lmcut", time_limit=20)
    assert tail.solved
    assert validate_plan(rerooted, tail.plan)


# ==========================================================================
# training
# ==========================================================================
def test_training_beats_predicting_the_mean(corpus):
    """The floor any regressor must clear to have learned anything."""
    bundle, report = train(corpus, TrainConfig(epochs=40, seed=0))
    targets = [s.target for s in corpus.samples]
    mean = sum(targets) / len(targets)
    baseline = sum(abs(t - mean) for t in targets) / len(targets)
    assert bundle.metrics["train_mae"] < baseline
    assert report.best_epoch >= 1
    assert report.history


def test_ranking_accuracy_is_reported_and_sane(corpus):
    bundle, _ = train(corpus, TrainConfig(epochs=40, seed=0))
    metrics = evaluate_ranking(bundle.model, corpus.groups, bundle.scale)
    assert 0.0 <= metrics["top1"] <= 1.0
    assert metrics["in_top2"] >= metrics["top1"]


def test_pure_regression_and_pure_ranking_both_train(corpus):
    for weight in (0.0, 1.0):
        bundle, _ = train(corpus, TrainConfig(epochs=15, rank_weight=weight, seed=0))
        assert math.isfinite(bundle.metrics["mae"])


def test_training_refuses_an_empty_corpus(space):
    with pytest.raises(ValueError):
        train(Corpus(space), TrainConfig(epochs=1))


def test_training_is_deterministic_given_a_seed(corpus):
    a, _ = train(corpus, TrainConfig(epochs=10, seed=3))
    b, _ = train(corpus, TrainConfig(epochs=10, seed=3))
    assert a.model.get_flat() == pytest.approx(b.model.get_flat())


# ==========================================================================
# the heuristic itself
# ==========================================================================
def test_bundle_round_trips_and_predicts_identically(corpus, ladder, tmp_path):
    bundle, _ = train(corpus, TrainConfig(epochs=20, seed=0))
    path = tmp_path / "model.heur.json"
    bundle.save(str(path))
    again = HeuristicBundle.load(str(path))
    _, task = ladder[0]
    state = task.initial_state()
    assert again.bind(task)(state) == pytest.approx(bundle.bind(task)(state))


def test_a_future_format_is_refused_rather_than_misread(corpus, tmp_path):
    bundle, _ = train(corpus, TrainConfig(epochs=5, seed=0))
    data = bundle.to_dict()
    data["format"] = 99
    path = tmp_path / "future.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        HeuristicBundle.load(str(path))


def test_the_heuristic_scores_a_goal_state_zero(corpus, ladder):
    bundle, _ = train(corpus, TrainConfig(epochs=20, seed=0))
    from jupyddl.api import solve_task

    _, task = ladder[0]
    result = solve_task(task, "astar", "lmcut", time_limit=20)
    goal = task.initial_state()
    for operator in result.plan:
        goal = task.apply(operator, goal)
    assert task.goal_reached(goal)
    assert bundle.bind(task)(goal) == 0.0


def test_the_heuristic_is_never_negative(corpus, ladder):
    bundle, _ = train(corpus, TrainConfig(epochs=20, seed=0))
    _, task = ladder[0]
    heuristic = bundle.bind(task)
    state = task.initial_state()
    assert heuristic(state) >= 0.0
    for operator in task.operators[:20]:
        if operator.applicable(state):
            assert heuristic(task.apply(operator, state)) >= 0.0


def test_values_are_cached_per_state(corpus, ladder):
    bundle, _ = train(corpus, TrainConfig(epochs=5, seed=0))
    _, task = ladder[0]
    heuristic = bundle.bind(task)
    state = task.initial_state()
    heuristic(state)
    heuristic(state)
    assert heuristic.evaluations == 1


def test_it_does_not_claim_to_be_admissible():
    """Nothing in the objective bounds the prediction from above."""
    assert LearnedHeuristic.admissible is False


# ==========================================================================
# integration: the registry, the API and the CLI all speak `learned:`
# ==========================================================================
def test_learned_spec_resolves_through_the_registry(corpus, ladder, tmp_path):
    bundle, _ = train(corpus, TrainConfig(epochs=10, seed=0))
    path = tmp_path / "reg.heur.json"
    bundle.save(str(path))
    _, task = ladder[0]
    heuristic = make_heuristic(f"learned:{path}", task)
    assert isinstance(heuristic, LearnedHeuristic)
    assert heuristic(task.initial_state()) >= 0.0


def test_an_already_built_heuristic_passes_through(ladder):
    _, task = ladder[0]
    built = make_heuristic("hff", task)
    assert make_heuristic(built, task) is built


def test_an_unknown_loader_is_rejected(ladder):
    _, task = ladder[0]
    with pytest.raises(ValueError, match="parameterised"):
        make_heuristic("psychic:model.json", task)


def test_solve_task_accepts_a_learned_spec(corpus, ladder, tmp_path):
    from jupyddl.api import solve_task, validate_plan

    bundle, _ = train(corpus, TrainConfig(epochs=20, seed=0))
    path = tmp_path / "solve.heur.json"
    bundle.save(str(path))
    _, task = ladder[-1]
    result = solve_task(task, "gbfs", f"learned:{path}", time_limit=20)
    assert result.solved
    assert validate_plan(
        task, result.plan
    ), "a learned heuristic must not break soundness"


def test_the_cli_accepts_and_rejects_heuristic_specs():
    from jupyddl.cli import heuristic_spec
    import argparse

    assert heuristic_spec("lmcut") == "lmcut"
    assert heuristic_spec("none") == "none"
    assert heuristic_spec("learned:x.json") == "learned:x.json"
    for bad in ("lmcutt", "learned:", "psychic:x"):
        with pytest.raises(argparse.ArgumentTypeError):
            heuristic_spec(bad)


def test_a_learned_heuristic_still_yields_valid_plans(corpus, ladder):
    """The heuristic may be wrong; the planner may not become unsound."""
    from jupyddl.api import solve_task, validate_plan

    bundle, _ = train(corpus, TrainConfig(epochs=20, seed=0))
    for _, task in ladder:
        result = solve_task(task, "gbfs", bundle.bind(task), time_limit=20)
        assert result.solved
        assert validate_plan(task, result.plan)


# ==========================================================================
# the reinforcement stages
# ==========================================================================
def test_search_cost_charges_a_failure_more_than_any_possible_success(corpus, ladder):
    """At a fixed budget, giving up must never look cheaper than finishing.

    Scores are only comparable *at the same budget* — the penalty is a multiple
    of it — so this compares a failure against the worst success the same
    budget allows, not against a run under a different one.
    """
    bundle, _ = train(corpus, TrainConfig(epochs=10, seed=0))
    config = RLConfig(max_expansions=1)  # nothing is solvable in one expansion
    starved = search_cost(bundle, ladder, config)
    assert starved.coverage == 0.0
    assert starved.score == pytest.approx(
        config.failure_penalty * config.max_expansions
    )
    assert starved.score > config.max_expansions


def test_search_cost_reports_full_coverage_when_everything_solves(corpus, ladder):
    bundle, _ = train(corpus, TrainConfig(epochs=10, seed=0))
    cost = search_cost(bundle, ladder, RLConfig(max_expansions=5000))
    assert cost.coverage == 1.0
    assert cost.score < 5000


def test_search_cost_needs_tasks(corpus):
    bundle, _ = train(corpus, TrainConfig(epochs=5, seed=0))
    with pytest.raises(ValueError):
        search_cost(bundle, [])


def test_dagger_adds_samples_from_the_search_distribution(corpus, ladder):
    bundle, _ = train(corpus, TrainConfig(epochs=10, seed=0))
    before = len(corpus)
    tuned, grown, history = dagger(
        bundle,
        ladder,
        corpus,
        rounds=1,
        states_per_task=4,
        config=RLConfig(max_expansions=500, seed=0),
        train_config=TrainConfig(epochs=5, seed=0),
    )
    assert len(grown) > before
    assert history[0]["round"] == 1
    assert any(s.instance.endswith("#dagger1") for s in grown.samples)
    # Aggregated labels come from a satisficing solver, so they are bounds.
    assert any(not s.optimal for s in grown.samples)
    assert tuned.bind(ladder[0][1])(ladder[0][1].initial_state()) >= 0.0


def test_bootstrap_records_what_it_newly_solved(corpus, ladder):
    bundle, _ = train(corpus, TrainConfig(epochs=10, seed=0))
    harder = tasks_from_generator("blocksworld", [5], seed=11, seeds_per_size=2)
    _, grown, history = bootstrap(
        bundle,
        harder,
        corpus,
        rounds=1,
        config=RLConfig(max_expansions=2000, seed=0),
        train_config=TrainConfig(epochs=5, seed=0),
    )
    assert history and "newly_solved" in history[0]
    assert history[0]["remaining"] + len(history[0]["newly_solved"]) == len(harder)


def test_cem_never_returns_a_worse_incumbent(corpus, ladder):
    """The incumbent is only displaced by a strictly better mean score."""
    bundle, _ = train(corpus, TrainConfig(epochs=15, seed=0))
    config = RLConfig(max_expansions=500, seed=0)
    before = search_cost(bundle, ladder, config).score
    tuned, history = optimise_search_cost(
        bundle, ladder, iterations=2, population=4, config=config
    )
    assert search_cost(tuned, ladder, config).score <= before
    assert history[0]["iteration"] == 0
    assert tuned.space == bundle.space


def test_cem_selects_the_incumbent_on_the_validation_set(corpus, ladder):
    """Tuning a thousand parameters against a handful of instances fits those
    instances. The returned model must improve on ones it was not fitted to."""
    bundle, _ = train(corpus, TrainConfig(epochs=15, seed=0))
    held_out = tasks_from_generator("blocksworld", [4], seed=321, seeds_per_size=2)
    config = RLConfig(max_expansions=800, seed=0)
    before = search_cost(bundle, held_out, config).score

    tuned, history = optimise_search_cost(
        bundle,
        ladder,
        iterations=2,
        population=4,
        config=config,
        validation_tasks=held_out,
    )
    # The reported score is the validation score, and the incumbent only ever
    # moves when that improves — so it cannot come back worse.
    assert search_cost(tuned, held_out, config).score <= before
    assert history[-1]["score"] <= before
    # Both scores are reported, so a run that is fitting its tuning set while
    # losing validation is visible rather than hidden.
    assert "tuning_score" in history[-1]


def test_cem_without_a_validation_set_scores_on_the_tuning_tasks(corpus, ladder):
    """The old behaviour stays reachable, and stays honest about what it means."""
    bundle, _ = train(corpus, TrainConfig(epochs=10, seed=0))
    config = RLConfig(max_expansions=500, seed=0)
    before = search_cost(bundle, ladder, config).score
    tuned, _ = optimise_search_cost(
        bundle, ladder, iterations=2, population=4, config=config
    )
    assert search_cost(tuned, ladder, config).score <= before


def test_cem_leaves_the_original_bundle_untouched(corpus, ladder):
    bundle, _ = train(corpus, TrainConfig(epochs=10, seed=0))
    before = list(bundle.model.get_flat())
    optimise_search_cost(
        bundle, ladder, iterations=1, population=4, config=RLConfig(max_expansions=300)
    )
    assert bundle.model.get_flat() == pytest.approx(before)


# ==========================================================================
# transfer
# ==========================================================================
def test_transfer_evaluation_reports_time_as_well_as_expansions(corpus, ladder):
    """Expansions alone can hide a heuristic that is slower per node than it
    is smart, which is the standard way to publish a non-result."""
    bundle, _ = train(corpus, TrainConfig(epochs=20, seed=0))
    rows = evaluate_transfer(
        bundle, ladder[:2], baselines=("goalcount",), max_expansions=2000
    )
    assert {row["heuristic"] for row in rows} == {"learned", "goalcount"}
    for row in rows:
        assert row["seconds"] >= 0.0
        assert "expanded" in row
    summary = summarise_transfer(rows)
    assert summary["learned"]["coverage"] == 1.0


def test_a_model_transfers_to_an_instance_far_larger_than_it_trained_on(corpus):
    """Not a quality claim — a claim that it runs at all, which is the thing
    a fixed-length one-hot encoding would have made impossible."""
    from jupyddl.api import solve_task, validate_plan

    bundle, _ = train(corpus, TrainConfig(epochs=30, seed=0))
    big = tasks_from_generator("blocksworld", [12], seed=99)[0][1]
    heuristic = bundle.bind(big)
    assert heuristic(big.initial_state()) >= 0.0
    result = solve_task(big, "gbfs", heuristic, max_expansions=20000, time_limit=30)
    if result.solved:
        assert validate_plan(big, result.plan)

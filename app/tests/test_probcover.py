"""
Tests for the ProbCover port (strategies/competitors/ProbCover.py).

The selection is a pure function over arrays, so these run without a trained model. The greedy cover is
checked against an independent plain-loop implementation, and the golden test compares it with the reference
code (dal-toolbox and the authors' copy) on tests/golden/probcover_golden.json, which
tests/golden/make_probcover_golden.py produces and is the only place the reference code runs.

    python app/tests/test_probcover.py    # or: pytest app/tests/test_probcover.py
"""

import json
import math
import os
import random
import sys
import traceback
import warnings
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
from torch.utils.data import Subset

from competitor_testing import build_strategy, config
from strategies.competitors.ProbCover import ProbCover, delta_from_purity, probcover_select


GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden', 'probcover_golden.json')


def unit_clusters(sizes, noise=0.15, dim=6, seed=0):
    """Clustered unit vectors, and the id of the cluster each one was drawn from."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(len(sizes), dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    points = np.concatenate([centre + noise * rng.normal(size=(size, dim)) for centre, size in zip(centres, sizes)])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points.astype(np.float32), np.repeat(np.arange(len(sizes)), sizes)


def safe_radius(points, low, high):
    """A radius in the widest gap between pairwise distances, so float32 rounding cannot flip an edge."""
    distances = np.sort(np.linalg.norm(points[:, None].astype(np.float64) - points[None], axis=-1)[np.triu_indices(len(points), 1)])
    inside = np.flatnonzero((distances[:-1] >= low) & (distances[1:] <= high))
    best = inside[np.diff(distances)[inside].argmax()]
    return float((distances[best] + distances[best + 1]) / 2)


def plain_greedy(points, labelled_rows, pool_rows, delta, budget):
    """Greedy max coverage with plain loops in float64. Returns the pool rows picked while some gain was positive."""
    points = points.astype(np.float64)
    near = lambda a, b: np.linalg.norm(points[a] - points[b]) < delta
    covered = {y for y in pool_rows if any(near(x, y) for x in labelled_rows)}
    balls = {c: {y for y in pool_rows if y == c or near(c, y)} for c in pool_rows}
    picks = []
    while len(picks) < budget:
        gains = {c: len(balls[c] - covered) for c in pool_rows if c not in picks}
        best = max(gains.values())
        if best == 0:
            break
        chosen = next(c for c in pool_rows if c not in picks and gains[c] == best)   # lowest position wins a tie
        covered |= balls[chosen]
        picks.append(chosen)
    return picks


def select(labelled, pool, budget, seed=0, **kwargs):
    kwargs.setdefault('n_classes', 5)
    kwargs.setdefault('l2_normalize', False)
    return probcover_select(labelled, pool, budget, np.random.default_rng(seed), **kwargs)


# ------------------------------------------------------------------ the radius
def test_purity_radius_is_the_largest_one_that_keeps_the_requested_purity():
    points, clusters = unit_clusters([30, 30, 30], noise=0.5, seed=1)       # overlapping clusters: purity < 1
    distances = np.linalg.norm(points[:, None].astype(np.float64) - points[None], axis=-1)

    def n_pure_balls(delta):
        return sum(all(clusters[y] == clusters[x] for y in range(len(points)) if distances[x, y] < delta) for x in range(len(points)))

    for alpha in (0.5, 0.8, 0.95, 1.0):
        needed = int(math.ceil(alpha * len(points) - 1e-9))
        delta = delta_from_purity(points, clusters, alpha)
        assert n_pure_balls(delta - 1e-5) >= needed, f'alpha {alpha}: delta {delta} is not pure enough'
        assert n_pure_balls(delta + 1e-5) < needed, f'alpha {alpha}: delta {delta} is not the largest radius'


def test_purity_radius_shrinks_as_the_threshold_grows():
    points, clusters = unit_clusters([30, 30, 30], noise=0.5, seed=1)
    deltas = [delta_from_purity(points, clusters, alpha) for alpha in (0.5, 0.8, 0.95, 1.0)]
    assert deltas == sorted(deltas, reverse=True) and deltas[0] > deltas[-1]


def test_purity_radius_with_a_single_cluster_is_finite_and_covers_everything():
    points, _ = unit_clusters([20])
    delta = delta_from_purity(points, np.zeros(20, dtype=int), 0.95)
    assert math.isfinite(delta) and delta >= 2.0 - 1e-6


def test_purity_radius_is_estimated_from_pseudo_labels_without_ground_truth():
    points, _ = unit_clusters([40, 40, 40], noise=0.1, seed=2)
    picks = select(points[:6], points[6:], 10, n_classes=3, delta=None)     # k-means with k = n_classes, no labels passed
    assert len(set(picks)) == 10


def test_the_radius_used_is_the_purity_radius_of_all_features_at_the_requested_threshold():
    module = sys.modules['strategies.competitors.ProbCover']
    points, clusters = unit_clusters([30, 30, 30], noise=0.5, seed=1)
    seen, real = {}, module._select_greedy_cover

    def spy(features, n_labelled, budget, delta, rng, block_size):
        seen['delta'] = delta
        return real(features, n_labelled, budget, delta, rng, block_size)

    deltas = []
    for alpha in (0.8, 0.95):
        with mock.patch.object(module, '_select_greedy_cover', spy), mock.patch.object(module, 'cluster_features', lambda *args: clusters):
            select(points[:6], points[6:], 5, n_classes=3, purity_threshold=alpha)
        assert seen['delta'] == delta_from_purity(points, clusters, alpha), alpha
        deltas.append(seen['delta'])
    assert deltas[0] > deltas[1], 'a stricter threshold must give a smaller radius'


def test_pseudo_label_k_means_uses_one_cluster_per_class_but_never_more_than_there_are_points():
    module = sys.modules['strategies.competitors.ProbCover']
    points, _ = unit_clusters([10, 10])
    requested, real = [], module.cluster_features

    def spy(features, n_clusters, *args):
        requested.append(n_clusters)
        return real(features, n_clusters, *args)

    with mock.patch.object(module, 'cluster_features', spy):
        select(points[:5], points[5:], 4, n_classes=3)
        select(points[:2], points[2:5], 2, n_classes=100)

    assert requested == [3, 5]


# ------------------------------------------------------------------ the greedy cover
def test_greedy_cover_matches_a_plain_implementation_whatever_the_block_size():
    for seed in range(4):
        points, _ = unit_clusters([14, 12, 10, 8, 6], noise=0.25, seed=seed)
        delta = safe_radius(points, 0.3, 0.5)
        labelled_rows = list(range(0, 50, 11))
        pool_rows = [row for row in np.random.default_rng(seed).permutation(len(points)) if row not in labelled_rows]
        expected = plain_greedy(points, labelled_rows, pool_rows, delta, 12)
        assert len(expected) >= 4, 'test precondition: the fixture must need several greedy picks'
        for block_size in (1000, 13, 7):
            picks = select(points[labelled_rows], points[pool_rows], 12, delta=delta, block_size=block_size)
            assert [pool_rows[pick] for pick in picks[:len(expected)]] == expected, f'seed {seed}, block {block_size}'


def test_graph_rows_beyond_the_last_full_block_are_not_dropped():
    # the reference loses the out-edges of the last n mod batch_size nodes: here the densest cluster is exactly there
    sparse, _ = unit_clusters([7, 7], noise=0.9, seed=3)
    dense = np.repeat(unit_clusters([1], seed=9)[0], 6, axis=0) + 1e-3 * np.random.default_rng(0).normal(size=(6, 6)).astype(np.float32)
    pool = np.concatenate([sparse, dense])
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)

    picks = select(np.empty((0, 6)), pool, 1, delta=0.05, block_size=7)

    assert picks[0] >= 14, 'the six near-duplicates at the end of the pool cover the most points'


def test_a_neighbour_at_exactly_delta_is_outside_the_ball():
    line = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    assert select(np.empty((0, 1)), line, 1, delta=1.0) == [0], 'strict inequality: every ball holds only its own centre'
    assert select(np.empty((0, 1)), line, 1, delta=1.0001) == [1], 'just above delta the middle point covers both neighbours'


def test_a_tiny_radius_still_lets_each_point_cover_itself():
    # large pools make torch.cdist use a matrix product, whose self-distance is not exactly 0
    features = np.random.default_rng(0).normal(size=(60, 512)).astype(np.float32)
    assert select(np.empty((0, 512)), features, 5, delta=1e-6, l2_normalize=True) == [0, 1, 2, 3, 4]


def test_a_candidate_at_exactly_delta_from_a_labelled_point_is_not_covered_by_it():
    # candidates 10, 10.5 | 1.0, 1.9, 2.8; the labelled point at 0 is exactly delta = 1 from the candidate at 1.0
    labelled = np.array([[0.0]], dtype=np.float32)
    pool = np.array([[10.0], [10.5], [1.0], [1.9], [2.8]], dtype=np.float32)
    # 1.9 covers {1.0, 1.9, 2.8} while 1.0 is uncovered; if it counted as covered, 10.0 would win the tie at gain 2
    assert select(labelled, pool, 1, delta=1.0) == [3]


def test_points_covered_by_the_labelled_set_are_not_worth_querying():
    points, clusters = unit_clusters([20, 15, 10], noise=0.1, seed=4)
    delta = safe_radius(points, 0.4, 0.7)
    labelled_rows = [0]                                                    # one labelled point inside cluster 0
    pool_rows = list(range(1, len(points)))

    picks = select(points[labelled_rows], points[pool_rows], 2, delta=delta)

    assert clusters[[pool_rows[pick] for pick in picks]].tolist() == [1, 2], 'cluster 0 is covered, the others are not'


# ------------------------------------------------------------------ contract
def test_selection_is_valid_for_every_budget_with_a_fixed_and_an_estimated_radius():
    points, _ = unit_clusters([30, 25, 20, 15, 10])
    labelled, pool = points[::7], np.delete(points, np.arange(0, 100, 7), axis=0)
    for delta in (0.35, None):
        for budget in (1, 5, 17, 40, len(pool)):
            picks = select(labelled, pool, budget, delta=delta)
            assert len(picks) == len(set(picks)) == budget, (delta, budget)
            assert min(picks) >= 0 and max(picks) < len(pool)
            assert all(type(pick) is int for pick in picks)


def test_budget_larger_than_the_pool_returns_the_whole_pool():
    points, _ = unit_clusters([12, 10, 8])
    for delta in (0.35, None):
        picks = select(points[:3], points[3:], 1000, delta=delta)
        assert sorted(picks) == list(range(len(points) - 3)), delta


def test_budget_zero_selects_nothing():
    points, _ = unit_clusters([10, 10])
    assert select(points[:3], points[3:], 0, delta=0.4) == []


def test_empty_labelled_set_is_supported():
    points, _ = unit_clusters([30, 25, 20])
    for delta in (0.35, None):
        assert len(set(select(np.empty((0, 6)), points, 20, delta=delta))) == 20


def test_once_everything_is_covered_the_rest_are_distinct_random_candidates():
    points, clusters = unit_clusters([20, 20, 20], noise=0.05, seed=5)
    labelled_rows = [0, 20, 40, 41]
    pool_rows = [row for row in range(len(points)) if row not in labelled_rows]
    kwargs = dict(delta=safe_radius(points, 0.6, 1.1))

    first = select(points[labelled_rows], points[pool_rows], 10, seed=1, **kwargs)
    second = select(points[labelled_rows], points[pool_rows], 10, seed=2, **kwargs)

    for picks in (first, second):
        assert len(picks) == len(set(picks)) == 10 and max(picks) < len(pool_rows)
    assert first != second, 'with nothing left to cover the picks come from the seeded generator'
    assert first == select(points[labelled_rows], points[pool_rows], 10, seed=1, **kwargs)


def test_same_generator_state_gives_the_same_selection_and_the_global_rngs_are_untouched():
    points, _ = unit_clusters([30, 30, 30], noise=0.4, seed=6)
    np.random.seed(5); random.seed(5); torch.manual_seed(5)
    before = (np.random.get_state()[1].copy(), random.getstate(), torch.get_rng_state().clone())

    first = probcover_select(points[:10], points[10:], 25, np.random.default_rng(11), n_classes=3, l2_normalize=False)
    second = probcover_select(points[:10], points[10:], 25, np.random.default_rng(11), n_classes=3, l2_normalize=False)

    after = (np.random.get_state()[1], random.getstate(), torch.get_rng_state())
    assert first == second
    assert (before[0] == after[0]).all() and before[1] == after[1] and (before[2] == after[2]).all()


def test_l2_normalisation_makes_the_selection_invariant_to_the_scale_of_each_row():
    points, _ = unit_clusters([20, 20, 20], noise=0.3, seed=7)
    scales = np.random.default_rng(0).uniform(0.5, 9.0, size=(60, 1)).astype(np.float32)
    delta = safe_radius(points, 0.3, 0.5)

    plain = select(points[:5], points[5:], 8, delta=delta, l2_normalize=True)
    rescaled = select(points[:5] * scales[:5], points[5:] * scales[5:], 8, delta=delta, l2_normalize=True)

    assert plain == rescaled
    assert select(points[:5] * scales[:5], points[5:] * scales[5:], 8, delta=delta, l2_normalize=False) != plain, \
        'test precondition: without normalisation the scale does change the picks'


def test_invalid_inputs_are_refused():
    points, clusters = unit_clusters([10, 10])
    with_nan = points.copy()
    with_nan[3, 1] = np.nan
    for call in (
        lambda: select(with_nan[:3], with_nan[3:], 2, delta=0.4),
        lambda: select(points[:3], points[3:], 2, delta=0.0),
        lambda: select(points[:3], points[3:], 2, delta=-1.0),
        lambda: select(points[:3], points[3:], 2, delta=float('nan')),
        lambda: select(points[:3], points[3:], 2, delta=float('inf')),
        lambda: delta_from_purity(points, clusters, 0.0),
        lambda: delta_from_purity(points, clusters, 1.5),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError('an invalid input was accepted')


# ------------------------------------------------------------------ strategy wrapper
def test_strategy_name_encodes_what_changes_the_numbers():
    assert build_strategy(ProbCover).strategy_name == 'ResNet_ProbCover'
    assert build_strategy(ProbCover, feature_source='file').strategy_name == 'ResNet_ProbCover_ssl'
    with config('probcover', delta=0.55):
        assert build_strategy(ProbCover).strategy_name == 'ResNet_ProbCover_d0.55'
        assert build_strategy(ProbCover, feature_source='file').strategy_name == 'ResNet_ProbCover_d0.55_ssl'
    with config('probcover', purity_threshold=0.9):
        assert build_strategy(ProbCover).strategy_name == 'ResNet_ProbCover_a0.9'


def test_query_maps_pool_positions_to_dataset_indices_and_avoids_the_labelled_set():
    points, _ = unit_clusters([30, 25, 20, 15], noise=0.3, seed=8)
    labelled_rows = np.arange(0, 90, 6)
    pool_rows = np.setdiff1d(np.arange(len(points)), labelled_rows)
    pool_indices = [int(index) for index in np.random.default_rng(0).permutation(np.arange(5000, 5000 + len(pool_rows)))]
    labelled_indices = [int(row) for row in labelled_rows]

    with config('probcover', l2_normalize=False):
        strategy = build_strategy(ProbCover)
    strategy.dataset, strategy.iter, strategy.labelled_indices = SimpleNamespace(dataset_id=1, n_classes=4), 2, labelled_indices
    strategy.get_features = lambda subset: (points[labelled_rows], points[pool_rows])

    local, global_indices = strategy.query(Subset(range(6000), pool_indices), 12)

    assert len(local) == len(global_indices) == len(set(global_indices)) == 12
    assert global_indices == [pool_indices[position] for position in local]
    assert not set(global_indices) & set(labelled_indices) and set(global_indices) <= set(pool_indices)
    assert strategy.query(Subset(range(6000), pool_indices), 12)[0] == local, 'the same (dataset, trial, cycle) must give the same query'


def test_query_with_a_pool_smaller_than_the_budget_returns_the_whole_pool():
    points, _ = unit_clusters([12, 10])
    strategy = build_strategy(ProbCover)
    strategy.dataset, strategy.iter, strategy.labelled_indices = SimpleNamespace(dataset_id=1, n_classes=2), 1, [0, 1, 2]
    strategy.get_features = lambda subset: (points[:3], points[3:])
    pool_indices = list(range(100, 100 + len(points) - 3))

    local, global_indices = strategy.query(Subset(range(200), pool_indices), 1000)

    assert sorted(local) == list(range(len(pool_indices))) and sorted(global_indices) == pool_indices


def test_probcover_is_registered():
    import main
    from init import dict_strategies_BB

    assert dict_strategies_BB['probcover'] is ProbCover
    argv = sys.argv
    try:
        sys.argv = ['main.py', '-m', 'probcover', '-ds', 'cifar10']
        assert main.get_args().methods == ['probcover']
    finally:
        sys.argv = argv


# ------------------------------------------------------------------ golden
def test_golden_matches_dal_toolbox_and_the_authors_code():
    with open(GOLDEN) as file:
        golden = json.load(file)
    features = np.asarray(golden['features'], dtype=np.float32)
    assert len(golden['scenarios']) >= 4
    for scenario in golden['scenarios']:
        labelled, pool = np.asarray(scenario['labelled'], dtype=int), np.asarray(scenario['pool'], dtype=int)
        picks = select(features[labelled], features[pool], scenario['budget'], delta=scenario['delta'])
        chosen = pool[picks].tolist()
        fixed = scenario['n_deterministic']

        assert chosen[:fixed] == scenario['expected'][:fixed], f'{scenario["name"]}: {chosen[:fixed]} != reference {scenario["expected"][:fixed]}'
        # past that point the references repeat one index (possibly a labelled sample); the port must not
        assert len(chosen) == len(set(chosen)) == scenario['budget'], scenario['name']
        assert not set(chosen) & set(scenario['labelled']), scenario['name']
        if fixed < scenario['budget']:
            assert scenario['reference_repeats_or_picks_labelled'], scenario['name']


if __name__ == "__main__":
    warnings.simplefilter('ignore')
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

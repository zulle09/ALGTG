"""
Tests for the TypiClust port (strategies/competitors/TypiClust.py).

The selection logic is a pure function over arrays, so these run without a trained model. The two
`golden` tests compare it with the reference code under an identical (injected) clustering; the
fixture in tests/golden/typiclust_golden.json is produced by tests/golden/make_typiclust_golden.py,
which is the only place the reference code is ever run.

    python app/tests/test_typiclust.py    # or: pytest app/tests/test_typiclust.py
"""

import json
import os
import random
import sys
import traceback
import warnings
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
from torch.utils.data import Subset

from competitor_config import competitor_params
from competitor_testing import build_strategy, config
from strategies.competitors.TypiClust import TypiClust, typicality, typiclust_select


GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden', 'typiclust_golden.json')


def structured_clusters(sizes, dim=8, seed=0):
    """Well-separated clusters whose density falls off with the distance to the centre, so the most
    typical point of a cluster is never a near-tie."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(len(sizes), dim)) * 20
    points, cluster_ids = [], []
    for cluster, (centre, size) in enumerate(zip(centres, sizes)):
        directions = rng.normal(size=(size, dim))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = 0.05 * (1 + np.arange(size)) ** 1.5
        points.append(centre + radii[:, None] * directions)
        cluster_ids.append(np.full(size, cluster))
    return np.concatenate(points).astype(np.float32), np.concatenate(cluster_ids)


def split(features, cluster_ids, labelled_rows):
    """(labelled features, pool features, labels of [labelled; pool], pool cluster ids)."""
    labelled_rows = np.asarray(labelled_rows, dtype=int)
    pool_rows = np.setdiff1d(np.arange(len(features)), labelled_rows)
    both = np.concatenate([labelled_rows, pool_rows])
    return features[labelled_rows], features[pool_rows], cluster_ids[both], cluster_ids[pool_rows]


def select(labelled, pool, budget, seed=0, **kwargs):
    kwargs.setdefault('l2_normalize', False)
    return typiclust_select(labelled, pool, budget, np.random.default_rng(seed), **kwargs)


# ------------------------------------------------------------------ typicality
def test_typicality_follows_the_reference_formula():
    points = np.array([[0.0], [1.0], [2.0], [3.5], [10.0]], dtype=np.float32)
    k = min(20, len(points) // 2)                      # 2 neighbours for 5 points
    expected = []
    for i in range(len(points)):
        squared = np.sort(((points - points[i]) ** 2).ravel())[1:k + 1]   # drop the point itself
        expected.append(1.0 / (squared.mean() + 1e-5))

    got = typicality(points, 20)

    assert np.allclose(got, expected, rtol=1e-5)
    assert got.argmax() == 1 and got.argmin() == 4, 'the densest point is x=1, the outlier x=10 the least typical'


def test_typicality_of_a_lone_point_is_infinite():
    assert np.isinf(typicality(np.zeros((1, 4), dtype=np.float32), 20)).all()


# ------------------------------------------------------------------ contract
def test_both_rules_return_distinct_in_range_indices_for_every_budget():
    features, cluster_ids = structured_clusters([30, 25, 20, 15, 10])
    labelled, pool, _, _ = split(features, cluster_ids, np.arange(0, 100, 7))
    for rule in ('authors', 'dal'):
        for budget in (1, 5, 17, 40, len(pool)):
            picks = select(labelled, pool, budget, selection=rule)
            assert len(picks) == budget, (rule, budget, len(picks))
            assert len(set(picks)) == budget, f'{rule}: duplicates for budget {budget}'
            assert min(picks) >= 0 and max(picks) < len(pool)
            assert all(type(pick) is int for pick in picks)


def test_budget_larger_than_the_pool_returns_the_whole_pool():
    features, cluster_ids = structured_clusters([12, 10, 8])
    labelled, pool, _, _ = split(features, cluster_ids, [0, 1, 2, 12, 13])
    for rule in ('authors', 'dal'):
        picks = select(labelled, pool, 1000, selection=rule)
        assert sorted(picks) == list(range(len(pool))), rule


def test_budget_zero_selects_nothing():
    features, _ = structured_clusters([10, 10])
    assert select(features[:3], features[3:], 0) == []


def test_empty_labelled_set_is_supported():
    features, _ = structured_clusters([30, 25, 20])
    for rule in ('authors', 'dal'):
        picks = select(np.empty((0, features.shape[1])), features, 20, selection=rule)
        assert len(set(picks)) == 20, rule


def test_same_generator_state_gives_the_same_selection_and_the_global_rngs_are_untouched():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(400, 16)).astype(np.float32)
    labelled, pool = features[:60], features[60:]
    for rule in ('authors', 'dal'):
        np.random.seed(5); random.seed(5); torch.manual_seed(5)
        before = (np.random.get_state()[1].copy(), random.getstate(), torch.get_rng_state().clone())

        first = typiclust_select(labelled, pool, 50, np.random.default_rng(11), selection=rule, max_clusters=40)
        second = typiclust_select(labelled, pool, 50, np.random.default_rng(11), selection=rule, max_clusters=40)

        after = (np.random.get_state()[1], random.getstate(), torch.get_rng_state())
        assert first == second, f'{rule} is not deterministic given the generator'
        assert (before[0] == after[0]).all() and before[1] == after[1] and (before[2] == after[2]).all(), \
            f'{rule} read or advanced a global RNG'


# ------------------------------------------------------------------ the two selection rules
def test_authors_rule_visits_clusters_by_fewest_labelled_then_largest_and_cycles():
    features, cluster_ids = structured_clusters([20, 15, 10, 8])
    labelled, pool, labels, pool_clusters = split(features, cluster_ids, [0, 1])   # two labelled points in cluster 0

    picks = select(labelled, pool, 6, selection='authors', cluster_labels=labels)

    # uncovered clusters by size (1, 2, 3), then the covered cluster 0, then round two: 1, 2
    assert pool_clusters[picks].tolist() == [1, 2, 3, 0, 1, 2]
    assert picks[0] != picks[4] and picks[1] != picks[5], 'a later visit must not repeat an earlier pick'
    for position, cluster in ((0, 1), (1, 2), (2, 3)):
        members = np.flatnonzero(pool_clusters == cluster)
        assert picks[position] == members[typicality(pool[members], 20).argmax()], 'not the most typical member'


def test_authors_rule_does_not_depend_on_the_generator():
    features, cluster_ids = structured_clusters([20, 15, 10, 8])
    labelled, pool, labels, _ = split(features, cluster_ids, [0, 1])
    results = {tuple(select(labelled, pool, 25, seed=seed, selection='authors', cluster_labels=labels)) for seed in range(4)}
    assert len(results) == 1, 'the authors rule has no randomness once the clustering is fixed'


def test_authors_rule_skips_exhausted_clusters_and_uses_dropped_clusters_last():
    # 3 clusters of 7 are kept (size > 5), 6 singletons are dropped. The official code crashes on the
    # second visit of an exhausted cluster; here every point must come out exactly once.
    features, cluster_ids = structured_clusters([7, 7, 7, 1, 1, 1, 1, 1, 1])
    labelled, pool, labels, pool_clusters = split(features, cluster_ids, [])

    picks = select(labelled, pool, len(pool), selection='authors', cluster_labels=labels, max_clusters=9)

    assert sorted(picks) == list(range(len(pool)))
    assert set(pool_clusters[picks[:21]]) == {0, 1, 2}, 'kept clusters first'
    assert set(pool_clusters[picks[21:]]) == {3, 4, 5, 6, 7, 8}, 'dropped clusters only once the kept ones are empty'


def test_dal_rule_takes_the_largest_uncovered_clusters_first_then_goes_random():
    features, cluster_ids = structured_clusters([20, 15, 10, 8])
    labelled, pool, labels, pool_clusters = split(features, cluster_ids, [0, 1])   # cluster 0 is covered

    typical_only = select(labelled, pool, 3, selection='dal', cluster_labels=labels)
    assert pool_clusters[typical_only].tolist() == [1, 2, 3]
    for position, cluster in enumerate((1, 2, 3)):
        members = np.flatnonzero(pool_clusters == cluster)
        assert typical_only[position] == members[typicality(pool[members], 20).argmax()]
    assert typical_only == select(labelled, pool, 3, seed=9, selection='dal', cluster_labels=labels)

    with_tail = select(labelled, pool, 8, selection='dal', cluster_labels=labels)
    assert with_tail[:3] == typical_only, 'the typical prefix does not depend on the budget'
    assert len(set(with_tail)) == 8


def test_dal_rule_is_all_random_when_every_cluster_holds_a_labelled_point():
    features, cluster_ids = structured_clusters([20, 15, 10, 8])
    labelled, pool, labels, _ = split(features, cluster_ids, [0, 20, 35, 45])      # one labelled point per cluster

    first = select(labelled, pool, 10, seed=1, selection='dal', cluster_labels=labels)
    second = select(labelled, pool, 10, seed=2, selection='dal', cluster_labels=labels)

    assert first != second and len(set(first)) == len(set(second)) == 10


# ------------------------------------------------------------------ features and clustering
def test_l2_normalisation_makes_the_selection_invariant_to_the_scale_of_each_row():
    rng = np.random.default_rng(0)
    features = np.abs(rng.normal(size=(300, 16))).astype(np.float32)
    labelled, pool = features[:40], features[40:]
    labels = np.random.default_rng(1).integers(0, 30, size=300)
    scales = rng.uniform(0.5, 9.0, size=(300, 1)).astype(np.float32)
    kwargs = dict(selection='authors', max_clusters=30, cluster_labels=labels, l2_normalize=True)

    plain = select(labelled, pool, 30, **kwargs)
    rescaled = select(labelled * scales[:40], pool * scales[40:], 30, **kwargs)

    assert plain == rescaled
    assert select(labelled * scales[:40], pool * scales[40:], 30, **{**kwargs, 'l2_normalize': False}) != plain, \
        'test precondition: without normalisation the scale does change the picks'


def test_identical_points_still_give_distinct_picks():
    features = np.ones((40, 8), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')                      # k-means warns that it found fewer clusters than asked
        for rule in ('authors', 'dal'):
            picks = select(features[:5], features[5:], 12, selection=rule)
            assert len(set(picks)) == 12, rule


def test_clusters_almost_as_numerous_as_points_work():
    # the smoke protocol: 64 labelled + 128 candidates, budget 64 -> K = 128 clusters over 192 points
    features = np.random.default_rng(0).normal(size=(192, 16)).astype(np.float32)
    for rule in ('authors', 'dal'):
        picks = select(features[:64], features[64:], 64, selection=rule)
        assert len(set(picks)) == 64, rule


def test_nan_features_are_refused():
    features = np.random.default_rng(0).normal(size=(50, 4)).astype(np.float32)
    features[7, 2] = np.nan
    try:
        select(features[:10], features[10:], 5)
    except ValueError:
        return
    raise AssertionError('NaN features must be refused, not silently clustered')


def test_bad_cluster_labels_and_selection_are_refused():
    features, cluster_ids = structured_clusters([10, 10])
    labelled, pool, labels, _ = split(features, cluster_ids, [0, 1])
    for bad in (labels[:-1], labels + 100, labels - 1):
        try:
            select(labelled, pool, 3, max_clusters=5, cluster_labels=bad)
        except ValueError:
            continue
        raise AssertionError('invalid cluster labels were accepted')
    try:
        select(labelled, pool, 3, selection='typical')
    except ValueError:
        return
    raise AssertionError('an unknown selection rule was accepted')


# ------------------------------------------------------------------ strategy wrapper
def test_strategy_name_encodes_what_changes_the_numbers():
    assert build_strategy(TypiClust).strategy_name == 'ResNet_TypiClust'
    assert build_strategy(TypiClust, feature_source='file').strategy_name == 'ResNet_TypiClust_ssl'
    with config('typiclust', selection='dal'):
        assert build_strategy(TypiClust).strategy_name == 'ResNet_TypiClust_dal'
        assert build_strategy(TypiClust, feature_source='file').strategy_name == 'ResNet_TypiClust_dal_ssl'


def test_max_clusters_follows_the_paper_per_dataset():
    assert build_strategy(TypiClust, 'cifar10').params['max_clusters'] == 500
    assert build_strategy(TypiClust, 'cifar100').params['max_clusters'] == 500
    assert build_strategy(TypiClust, 'tinyimagenet').params['max_clusters'] == 1000
    assert 'max_clusters_by_dataset' not in build_strategy(TypiClust).params
    assert 'max_clusters_by_dataset' in competitor_params['typiclust'], 'the shared config must not be modified'


def test_query_maps_pool_positions_to_dataset_indices_and_avoids_the_labelled_set():
    features, cluster_ids = structured_clusters([30, 25, 20, 15])
    labelled_rows = np.arange(0, 90, 6)
    labelled, pool, _, _ = split(features, cluster_ids, labelled_rows)
    pool_indices = [int(index) for index in np.random.default_rng(0).permutation(np.arange(5000, 5000 + len(pool)))]
    labelled_indices = [int(row) for row in labelled_rows]

    with config('typiclust', l2_normalize=False):
        strategy = build_strategy(TypiClust)
    strategy.dataset, strategy.iter, strategy.labelled_indices = SimpleNamespace(dataset_id=1), 2, labelled_indices
    strategy.get_features = lambda subset: (labelled, pool)

    local, global_indices = strategy.query(Subset(range(6000), pool_indices), 12)

    assert len(local) == len(global_indices) == len(set(global_indices)) == 12
    assert global_indices == [pool_indices[position] for position in local]
    assert not set(global_indices) & set(labelled_indices)
    assert set(global_indices) <= set(pool_indices)
    assert strategy.query(Subset(range(6000), pool_indices), 12)[0] == local, \
        'the same (dataset, trial, cycle) must give the same query'


def test_query_with_a_pool_smaller_than_the_budget_returns_the_whole_pool():
    features, cluster_ids = structured_clusters([12, 10])
    labelled, pool, _, _ = split(features, cluster_ids, [0, 1, 2])
    pool_indices = list(range(100, 100 + len(pool)))

    strategy = build_strategy(TypiClust)
    strategy.dataset, strategy.iter, strategy.labelled_indices = SimpleNamespace(dataset_id=1), 1, [0, 1, 2]
    strategy.get_features = lambda subset: (labelled, pool)

    local, global_indices = strategy.query(Subset(range(200), pool_indices), 1000)

    assert sorted(local) == list(range(len(pool))) and sorted(global_indices) == pool_indices


def test_typiclust_is_registered_and_the_feature_flags_fail_fast():
    import main
    from init import dict_strategies_BB

    assert dict_strategies_BB['typiclust'] is TypiClust
    argv = sys.argv
    try:
        sys.argv = ['main.py', '-m', 'typiclust', '-ds', 'cifar10']
        args = main.get_args()
        assert args.methods == ['typiclust'] and args.feature_source == 'model' and args.feature_path is None

        sys.argv = ['main.py', '-m', 'typiclust', '-ds', 'cifar10', '-fs', 'file', '-fp', '/does/not/exist.npy']
        try:
            main.get_args()
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError('-fs file with a missing -fp file must be refused before any training starts')
    finally:
        sys.argv = argv


# ------------------------------------------------------------------ golden
def run_golden(rule):
    with open(GOLDEN) as file:
        golden = json.load(file)
    features = np.asarray(golden['features'], dtype=np.float32)
    cluster_ids = np.asarray(golden['cluster_labels'])
    checked = 0
    for scenario in golden['scenarios']:
        if scenario['rule'] != rule:
            continue
        labelled, pool = np.asarray(scenario['labelled'], dtype=int), np.asarray(scenario['pool'], dtype=int)
        picks = select(
            features[labelled], features[pool], scenario['budget'], selection=rule,
            max_clusters=scenario['max_clusters'], cluster_labels=cluster_ids[np.concatenate([labelled, pool])]
        )
        chosen = pool[picks].tolist()
        fixed = scenario['n_deterministic']
        assert chosen[:fixed] == scenario['expected'][:fixed], \
            f'{scenario["name"]}: {chosen[:fixed]} != reference {scenario["expected"][:fixed]}'
        # what follows the deterministic prefix is drawn from a different generator: only invariants hold
        assert len(chosen) == len(set(chosen)) == len(scenario['expected']), scenario['name']
        assert not set(chosen) & set(scenario['labelled']), scenario['name']
        checked += 1
    assert checked > 0, f'no golden scenario for the {rule} rule'


def test_golden_dal_rule_matches_dal_toolbox():
    run_golden('dal')


def test_golden_authors_rule_matches_the_official_code():
    run_golden('authors')


if __name__ == "__main__":
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

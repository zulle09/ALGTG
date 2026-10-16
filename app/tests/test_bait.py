"""
Tests for the BAIT port (strategies/competitors/BAIT.py).

The port tracks, for every candidate, small (rank x rank) matrices that it updates with rank-r corrections,
and never builds the (rank x C*D) embeddings. So the central test compares it with `dense_bait`, which keeps
explicit matrices and re-evaluates the trace objective with an explicit inverse at every step, and the golden
test compares it with dal-toolbox's BaitSampling on tests/golden/bait_golden.json, produced by
tests/golden/make_bait_golden.py, the only place the reference code runs.

    python app/tests/test_bait.py                         # or: pytest app/tests/test_bait.py
    BAIT_TEST_DEVICE=cuda python app/tests/test_bait.py   # the same tests on a GPU
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

from competitor_testing import build_strategy, config
from golden.dense_bait import dense_bait, explicit_embeddings, softmax
from strategies.competitors.BAIT import BAIT, BAIT_bce, bait_select, fisher_factors, fisher_sum, project, quadratic_forms


DEVICE = os.environ.get('BAIT_TEST_DEVICE', 'cpu')
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden', 'bait_golden.json')


def make_inputs(n_labelled, n_pool=22, n_classes=4, dim=5, seed=0, scale=1.5):
    """(labelled logits, labelled features, pool logits, pool features); features are non-negative, as after a ReLU."""
    rng = np.random.default_rng(seed)
    features = np.abs(rng.normal(size=(n_labelled + n_pool, dim)))
    logits = rng.normal(scale=scale, size=(n_labelled + n_pool, n_classes))
    return logits[:n_labelled], features[:n_labelled], logits[n_labelled:], features[n_labelled:]


def select(inputs, budget, **kwargs):
    kwargs.setdefault('device', DEVICE)
    return bait_select(*inputs, budget, **kwargs)


tensor = lambda array: torch.as_tensor(np.asarray(array), dtype=torch.float64, device=DEVICE)


# ------------------------------------------------------------------ against the definition
def test_selection_matches_the_dense_objective():
    for kind in ('ce', 'bce'):
        for n_labelled in (0, 9):
            for seed in range(3):
                inputs = make_inputs(n_labelled, seed=seed)
                expected, gaps = dense_bait(*inputs, 4, 0.3, kind)
                assert min(gaps) > 1e-6, 'test precondition: no near-tie in the fixture'
                assert select(inputs, 4, fisher=kind, lambda_reg=0.3) == expected, (kind, n_labelled, seed)


def test_selection_matches_the_dense_objective_for_other_oversampling_and_budgets():
    inputs = make_inputs(7, n_pool=22, seed=5)
    for budget, oversample in ((14, 2), (6, 1), (3, 3), (5, 4)):    # 2 * 14 > 22 exhausts the pool before pruning
        for kind in ('ce', 'bce'):
            expected, gaps = dense_bait(*inputs, budget, 0.5, kind, oversample)
            assert min(gaps) > 1e-6
            assert select(inputs, budget, fisher=kind, lambda_reg=0.5, oversample=oversample) == expected, (budget, oversample, kind)


def test_fisher_factors_reproduce_the_softmax_fisher():
    rng = np.random.default_rng(0)
    logits, features = rng.normal(scale=2.0, size=(6, 4)), np.abs(rng.normal(size=(6, 5)))
    probs = softmax(logits)
    ce, bce = fisher_factors(logits, 'ce'), fisher_factors(logits, 'bce')
    assert ce.shape == (6, 4, 4) and bce.shape == (6, 2, 1)

    for i in range(6):
        expected_ce = np.kron(np.diag(probs[i]) - np.outer(probs[i], probs[i]), np.outer(features[i], features[i]))
        top = probs[i].max()
        expected_bce = top * (1 - top) * np.outer(features[i], features[i])
        assert np.allclose(fisher_sum(tensor(ce[i:i + 1]), tensor(features[i:i + 1])).cpu().numpy(), expected_ce)
        assert np.allclose(fisher_sum(tensor(bce[i:i + 1]), tensor(features[i:i + 1])).cpu().numpy(), expected_bce)
        assert np.allclose(np.kron(ce[i], features[i][None, :]), explicit_embeddings(logits[i:i + 1], features[i:i + 1], 'ce')[0])
        assert np.allclose(np.kron(bce[i], features[i][None, :]), explicit_embeddings(logits[i:i + 1], features[i:i + 1], 'bce')[0])

    summed = sum(np.kron(np.diag(p) - np.outer(p, p), np.outer(f, f)) for p, f in zip(probs, features))
    assert np.allclose(fisher_sum(tensor(ce), tensor(features)).cpu().numpy(), summed)


def test_projection_and_quadratic_forms_match_explicit_embeddings():
    rng = np.random.default_rng(1)
    logits, features = rng.normal(scale=2.0, size=(7, 3)), np.abs(rng.normal(size=(7, 4)))
    for kind in ('ce', 'bce'):
        explicit = explicit_embeddings(logits, features, kind)
        factors = tensor(fisher_factors(logits, kind))
        dimension = explicit.shape[2]
        columns, symmetric = rng.normal(size=(dimension, 3)), rng.normal(size=(dimension, dimension))
        symmetric = symmetric + symmetric.T
        assert np.allclose(project(factors, tensor(features), tensor(columns)).cpu().numpy(), explicit @ columns)
        assert np.allclose(quadratic_forms(tensor(symmetric), factors, tensor(features)).cpu().numpy(), explicit @ symmetric @ explicit.transpose(0, 2, 1))


# ------------------------------------------------------------------ contract
def test_selection_is_valid_for_every_budget():
    inputs = make_inputs(6)
    for kind in ('ce', 'bce'):
        for budget in (1, 3, 7, 11, 21):
            picks = select(inputs, budget, fisher=kind)
            assert len(picks) == len(set(picks)) == budget, (kind, budget)
            assert min(picks) >= 0 and max(picks) < 22 and all(type(pick) is int for pick in picks)


def test_budget_equal_to_or_larger_than_the_pool_returns_the_whole_pool():
    inputs = make_inputs(4, n_pool=10)
    for budget in (10, 100):
        assert select(inputs, budget) == list(range(10)), 'nothing to choose: the pool comes back in index order'


def test_budget_zero_selects_nothing():
    assert select(make_inputs(4), 0) == []


def test_empty_labelled_set_accepts_empty_arrays_of_any_shape():
    _, _, pool_logits, pool_features = make_inputs(0)
    picks = select((np.empty((0,)), np.empty((0, 5)), pool_logits, pool_features), 4)
    assert picks == select(make_inputs(0), 4)


def test_selection_is_deterministic_and_never_touches_the_global_rngs():
    inputs = make_inputs(8, seed=2)
    np.random.seed(5); random.seed(5); torch.manual_seed(5)
    before = (np.random.get_state()[1].copy(), random.getstate(), torch.get_rng_state().clone())

    first, second = select(inputs, 5), select(inputs, 5)

    after = (np.random.get_state()[1], random.getstate(), torch.get_rng_state())
    assert first == second
    assert (before[0] == after[0]).all() and before[1] == after[1] and (before[2] == after[2]).all()


def test_candidates_with_saturated_probabilities_are_finite_and_picked_last():
    labelled_logits, labelled_features, pool_logits, pool_features = make_inputs(6, seed=3)
    pool_logits = pool_logits.copy()
    pool_logits[[2, 9]] = [[40.0, 0, 0, 0], [0, 0, 40.0, 0]]        # an overfit network is certain about these two
    for kind in ('ce', 'bce'):
        picks = select((labelled_logits, labelled_features, pool_logits, pool_features), 5, fisher=kind)
        assert len(set(picks)) == 5 and not {2, 9} & set(picks), kind


def test_exact_duplicates_follow_the_tie_rules():
    # six identical candidates: forward picks the lowest unselected index each time (0, 1, 2, 3), the
    # backward pass removes the first of the tied members of the selection each time (0, then 1)
    logits, features = np.tile([[1.0, 0.2, -0.5, 0.0]], (6, 1)), np.tile([[1.0, 2.0, 0.5, 1.5, 0.3]], (6, 1))
    assert select((np.empty((0, 4)), np.empty((0, 5)), logits, features), 2, fisher='ce') == [2, 3]


def test_invalid_inputs_are_refused():
    labelled_logits, labelled_features, pool_logits, pool_features = make_inputs(4)
    bad_pool = pool_features.copy(); bad_pool[3, 1] = np.nan
    bad_logits = pool_logits.copy(); bad_logits[5, 0] = np.inf
    bad_labelled = labelled_features.copy(); bad_labelled[0, 0] = np.nan
    for call in (
        lambda: select((labelled_logits, labelled_features, pool_logits, bad_pool), 3),
        lambda: select((labelled_logits, labelled_features, bad_logits, pool_features), 3),
        lambda: select((labelled_logits, bad_labelled, pool_logits, pool_features), 3),
        lambda: select(make_inputs(4), 3, lambda_reg=0.0),
        lambda: select(make_inputs(4), 3, lambda_reg=-1.0),
        lambda: select(make_inputs(4), 3, fisher='full'),
        lambda: select(make_inputs(4), 3, oversample=0),
        lambda: select(make_inputs(4), 3, max_embedding_dim=10),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError('an invalid input was accepted')


def test_a_fisher_matrix_that_does_not_fit_is_refused_with_the_way_out():
    try:
        select(make_inputs(4), 3, max_embedding_dim=10)
    except ValueError as error:
        assert 'bce' in str(error) and '20 x 20' in str(error)
        return
    raise AssertionError('the dimension limit was not enforced')


# ------------------------------------------------------------------ strategy wrapper
def make_strategy(dataset_name='cifar10', n_classes=10, embedding_dim=512, strategy_class=BAIT, **ct_p):
    model = SimpleNamespace(backbone=SimpleNamespace(get_embedding_dim=lambda: embedding_dim))
    return build_strategy(strategy_class, dataset_name, Dataset=SimpleNamespace(n_classes=n_classes), Master_Model=model, **ct_p)


def test_strategy_name_encodes_what_changes_the_numbers():
    assert make_strategy('cifar10').strategy_name == 'ResNet_BAIT'            # the paper's lambda differs by dataset
    assert make_strategy('svhn').strategy_name == 'ResNet_BAIT'
    assert make_strategy('cifar100', n_classes=100, strategy_class=BAIT_bce).strategy_name == 'ResNet_BAIT_bce'
    assert make_strategy('svhn', strategy_class=BAIT_bce).strategy_name == 'ResNet_BAIT_bce'
    with config('bait', lambda_reg=0.5):
        assert make_strategy('svhn').strategy_name == 'ResNet_BAIT_l0.5'
        assert make_strategy('cifar10').strategy_name == 'ResNet_BAIT'        # cifar10 keeps its own value
    with config('bait', lambda_by_dataset={}, lambda_reg=0.5):
        assert make_strategy('cifar10').strategy_name == 'ResNet_BAIT_l0.5'
    with config('bait', oversample=3):
        assert make_strategy().strategy_name == 'ResNet_BAIT_o3'


def test_lambda_follows_the_paper_per_dataset():
    assert make_strategy('cifar10').params['lambda_reg'] == 0.01
    assert make_strategy('svhn').params['lambda_reg'] == 1.0
    assert make_strategy('fmnist').params['lambda_reg'] == 1.0
    assert 'lambda_by_dataset' not in make_strategy().params


def test_features_from_a_file_are_refused():
    try:
        make_strategy(feature_source='file')
    except ValueError as error:
        assert 'logits' in str(error)
        return
    raise AssertionError('BAIT needs the trained model; -fs file must be refused')


def test_an_infeasible_fisher_matrix_is_refused_at_construction():
    try:
        make_strategy('cifar100', n_classes=100)
    except ValueError as error:
        assert 'bce' in str(error)
    else:
        raise AssertionError('51200 x 51200 doubles must be refused before any training starts')
    assert make_strategy('cifar100', n_classes=100, strategy_class=BAIT_bce).params['fisher'] == 'bce'
    assert make_strategy().params['fisher'] == 'ce'


def prepare_query(strategy, labelled, pool):
    """Give the strategy what ActiveLearner would: the model outputs of the labelled set, then of the pool."""
    outputs = iter([(labelled[1], labelled[0]), (pool[1], pool[0])])
    strategy._outputs = lambda subset: next(outputs)
    strategy.load_best_checkpoint = lambda: None
    strategy.dataset = SimpleNamespace(dataset_id=1, n_classes=4, unlab_train_ds=range(10_000))


def test_query_maps_pool_positions_to_dataset_indices_and_avoids_the_labelled_set():
    labelled_logits, labelled_features, pool_logits, pool_features = make_inputs(8, n_pool=30, seed=4)
    pool_indices = [int(index) for index in np.random.default_rng(0).permutation(np.arange(5000, 5030))]
    labelled_indices = list(range(8))

    with config('bait', device=DEVICE, max_embedding_dim=100):
        strategy = make_strategy(embedding_dim=5, n_classes=4)
    strategy.labelled_indices = labelled_indices
    prepare_query(strategy, (labelled_logits, labelled_features), (pool_logits, pool_features))

    local, global_indices = strategy.query(Subset(range(6000), pool_indices), 5)

    assert len(local) == len(global_indices) == len(set(global_indices)) == 5
    assert global_indices == [pool_indices[position] for position in local]
    assert not set(global_indices) & set(labelled_indices) and set(global_indices) <= set(pool_indices)
    assert strategy.params['lambda_reg'] == 0.01, 'test precondition: cifar10 uses the paper value'
    assert local == bait_select(labelled_logits, labelled_features, pool_logits, pool_features, 5, **strategy.params)
    assert local != bait_select(labelled_logits, labelled_features, pool_logits, pool_features, 5, device=DEVICE), \
        'test precondition: lambda must matter for this fixture, otherwise the parameters are not being checked'


def test_query_with_a_pool_smaller_than_the_budget_returns_the_whole_pool():
    labelled_logits, labelled_features, pool_logits, pool_features = make_inputs(3, n_pool=7)
    strategy = make_strategy(embedding_dim=5, n_classes=4)
    strategy.labelled_indices = [0, 1, 2]
    prepare_query(strategy, (labelled_logits, labelled_features), (pool_logits, pool_features))
    pool_indices = list(range(100, 107))

    local, global_indices = strategy.query(Subset(range(200), pool_indices), 1000)

    assert sorted(local) == list(range(7)) and sorted(global_indices) == pool_indices


def test_bait_is_registered():
    import main
    from init import dict_strategies_BB

    assert dict_strategies_BB['bait'] is BAIT and dict_strategies_BB['bait_bce'] is BAIT_bce
    argv = sys.argv
    try:
        sys.argv = ['main.py', '-m', 'bait', 'bait_bce', '-ds', 'cifar10']
        assert main.get_args().methods == ['bait', 'bait_bce']
    finally:
        sys.argv = argv


# ------------------------------------------------------------------ golden
def test_golden_matches_dal_toolbox():
    with open(GOLDEN) as file:
        golden = json.load(file)
    features, logits = np.asarray(golden['features']), np.asarray(golden['logits'])
    assert len(golden['scenarios']) >= 4
    for scenario in golden['scenarios']:
        labelled, pool = np.asarray(scenario['labelled'], dtype=int), np.asarray(scenario['pool'], dtype=int)
        picks = select((logits[labelled], features[labelled], logits[pool], features[pool]), scenario['budget'],
                       fisher=scenario['fisher'], lambda_reg=scenario['lambda_reg'])
        assert pool[picks].tolist() == scenario['expected'], f'{scenario["name"]}: {pool[picks].tolist()} != reference {scenario["expected"]}'


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

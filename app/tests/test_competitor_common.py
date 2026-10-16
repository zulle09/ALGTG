"""
Tests for the helpers shared by the ported competitor strategies (strategies/competitors/common.py).

    python app/tests/test_competitor_common.py    # or: pytest app/tests/test_competitor_common.py
"""

import os
import random
import sys
import tempfile
import traceback
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from strategies.competitors.common import FeatureSourceMixin, check_selection, derive_rng, load_feature_table


class ToyDataset(Dataset):
    """Item i is (i, a 3-d 'image' filled with i, label 0): easy to check which rows came back."""

    def __init__(self, n_samples: int = 50):
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return index, torch.full((3,), float(index)), 0


class FakeStrategy(FeatureSourceMixin):
    """The part of ActiveLearner the mixin touches, with a toy 'model' that embeds an image as itself."""

    def __init__(self, ct_p, labelled_indices):
        self.ct_p = ct_p
        self.labelled_indices = labelled_indices
        self.dataset = SimpleNamespace(train_ds=ToyDataset(), unlab_train_ds=ToyDataset())
        self.batch_size = 4
        self.model = SimpleNamespace(backbone=SimpleNamespace(get_embedding_dim=lambda: 3))
        self.checkpoint_loads = 0

    def load_best_checkpoint(self):
        self.checkpoint_loads += 1

    def get_embeddings(self, dataloader, dict_to_modify, embedds2cpu=False):
        for _, images, _ in dataloader:
            dict_to_modify['embedds'] = torch.cat((dict_to_modify['embedds'], images))


def test_derive_rng_is_a_pure_function_of_its_arguments():
    draw = lambda *args: derive_rng(*args).integers(0, 10 ** 9, 8)
    assert (draw(1, 0, 2, 'ResNet_X') == draw(1, 0, 2, 'ResNet_X')).all()
    for other in [(2, 0, 2, 'ResNet_X'), (1, 1, 2, 'ResNet_X'), (1, 0, 3, 'ResNet_X'), (1, 0, 2, 'ResNet_Y')]:
        assert not (draw(*other) == draw(1, 0, 2, 'ResNet_X')).all(), f'{other} gave the same stream'


def test_derive_rng_leaves_the_global_generators_alone():
    np.random.seed(7); random.seed(7); torch.manual_seed(7)
    before = (np.random.get_state()[1].copy(), random.getstate(), torch.get_rng_state().clone())
    derive_rng(1, 0, 1, 'ResNet_X').integers(0, 10, 100)
    after = (np.random.get_state()[1], random.getstate(), torch.get_rng_state())
    assert (before[0] == after[0]).all() and before[1] == after[1] and (before[2] == after[2]).all()


def test_check_selection_accepts_valid_selections():
    assert check_selection([3, 1, 2], 10, 3) == [3, 1, 2]
    assert check_selection(np.array([0, 4]), 5, 2) == [0, 4]
    assert check_selection(range(4), 4, 10) == [0, 1, 2, 3], 'a budget larger than the pool selects the whole pool'
    assert check_selection([], 0, 5) == []


def test_check_selection_refuses_what_update_sets_cannot_use():
    for selected, n_candidates, budget in [([1, 1, 2], 10, 3), ([1, 2], 10, 3), ([1, 2, 10], 10, 3), ([1, 2, -1], 10, 3), ([0, 1, 2, 3], 10, 3)]:
        try:
            check_selection(selected, n_candidates, budget)
        except ValueError:
            continue
        raise AssertionError(f'{selected} (pool {n_candidates}, budget {budget}) should have been refused')


def test_load_feature_table_validates_shape_and_values():
    with tempfile.TemporaryDirectory() as folder:
        good, wrong_rows, wrong_rank, with_nan = (os.path.join(folder, f'{name}.npy') for name in ('good', 'rows', 'rank', 'nan'))
        np.save(good, np.ones((50, 4), dtype=np.float64))
        np.save(wrong_rows, np.ones((49, 4)))
        np.save(wrong_rank, np.ones(50))
        np.save(with_nan, np.full((50, 4), np.nan))

        table = load_feature_table(good, 50)
        assert table.shape == (50, 4) and table.dtype == np.float32
        for bad in (wrong_rows, wrong_rank, with_nan):
            try:
                load_feature_table(bad, 50)
            except ValueError:
                continue
            raise AssertionError(f'{bad} should have been refused')


def test_file_source_returns_the_rows_of_the_requested_dataset_indices():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, 'features.npy')
        np.save(path, np.repeat(np.arange(50, dtype=np.float32)[:, None], 3, axis=1))
        strategy = FakeStrategy({'feature_source': 'file', 'feature_path': path}, labelled_indices=[3, 7])

        labelled, pool = strategy.get_features(Subset(strategy.dataset.unlab_train_ds, [10, 5, 20]))

        assert labelled[:, 0].tolist() == [3.0, 7.0]
        assert pool[:, 0].tolist() == [10.0, 5.0, 20.0], 'the pool rows must follow the Subset order'
        assert strategy.checkpoint_loads == 0, 'precomputed features do not need the trained model'


def test_file_source_with_an_empty_labelled_set():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, 'features.npy')
        np.save(path, np.ones((50, 3), dtype=np.float32))
        strategy = FakeStrategy({'feature_source': 'file', 'feature_path': path}, labelled_indices=[])

        labelled, pool = strategy.get_features(Subset(strategy.dataset.unlab_train_ds, [1, 2]))

        assert labelled.shape == (0, 3) and pool.shape == (2, 3)


def test_model_source_embeds_labelled_and_pool_in_order_after_loading_the_checkpoint():
    strategy = FakeStrategy({}, labelled_indices=[9, 2, 30])

    labelled, pool = strategy.get_features(Subset(strategy.dataset.unlab_train_ds, [11, 4, 40, 8, 1]))

    assert labelled[:, 0].tolist() == [9.0, 2.0, 30.0]
    assert pool[:, 0].tolist() == [11.0, 4.0, 40.0, 8.0, 1.0]
    assert labelled.dtype == np.float32 and labelled.shape == (3, 3)
    assert strategy.checkpoint_loads == 1


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

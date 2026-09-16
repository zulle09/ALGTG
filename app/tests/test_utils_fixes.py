"""Regression tests for the bugs fixed in `app/utils.py`.

Each test names the failure it locks out. They are fast (no training, no real dataset) and are
meant to be run before a sweep together with the other test modules:

    python app/tests/test_utils_fixes.py
"""

import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import matplotlib
matplotlib.use('Agg')  # no display on a headless GPU box
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import matplotlib.figure
from scipy import stats

from utils import (
    MARKERS, _is_random_method, count_class_observation, entropy, get_palette,
    plot_gtg_entropy_tensor, plot_res_std_mean, plot_trail_acc, purge_rows, read_rows,
    save_train_val_curves, set_seeds, write_csv,
)


class _FakeDS(Dataset):
    """Minimal (index, image, label) dataset, the shape `count_class_observation` expects."""

    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return i, torch.zeros(3, 4, 4), int(self.labels[i])


def test_random_dashed_is_equality_not_substring():
    """`split('_')[1] in 'Random'` matched any substring of "Random" and crashed without a '_'."""
    assert _is_random_method('ResNet18_Random')
    assert not _is_random_method('ResNet18_GTG_off_1_0')
    assert not _is_random_method('ResNet18_TA_VAAL')
    # substrings of "Random" that the old test accepted
    assert not _is_random_method('ResNet18_and')
    assert not _is_random_method('ResNet18_do')
    assert not _is_random_method('ResNet18_')
    # no underscore at all: used to raise IndexError
    assert not _is_random_method('Random')


def test_palette_does_not_disturb_the_experiment_rng():
    """Plotting must not consume the global RNG that `set_seeds` seeds.

    Otherwise a run with `-diag` draws a different unlabelled sub-sample than one without, and
    the two are no longer comparable.
    """
    set_seeds(42)
    without_plotting = [random.random() for _ in range(5)]

    set_seeds(42)
    for n in (10, 50, 200):  # every palette branch
        get_palette(n)
    with_plotting = [random.random() for _ in range(5)]

    assert without_plotting == with_plotting


def test_palette_is_still_shuffled_and_deterministic():
    a, b = get_palette(10), get_palette(10)
    assert sorted(a) == sorted(b) and len(a) == 10


def test_figures_are_closed():
    """Every plotting helper leaked a figure; over a sweep matplotlib grows without bound."""
    plt.close('all')
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            os.makedirs('results/exp/cifar10/0', exist_ok=True)
            n_open_before = len(plt.get_fignums())

            for _ in range(5):
                plot_trail_acc(
                    dataset_name='cifar10', trial=0,
                    methods_results={f'ResNet18_{m}': {'test_accuracy': [1., 2., 3.]}
                                     for m in ['Random', 'Entropy', 'GTG_off']},
                    n_lab_obs=[100, 200, 300], exp_path='exp', key='test_accuracy',
                    plot_png_name='cifar10/0/results.png',
                )

            assert len(plt.get_fignums()) == n_open_before, \
                f'{len(plt.get_fignums()) - n_open_before} figures leaked'
        finally:
            os.chdir(cwd)


def test_plot_trail_acc_rejects_missing_filename():
    """The docstring promised a ValueError; the code wrote to `results/<exp>/None` instead."""
    try:
        plot_trail_acc('cifar10', 0, {}, [], 'exp', 'test_accuracy', save_plot=True, plot_png_name=None)
    except ValueError:
        return
    raise AssertionError('expected ValueError when plot_png_name is None')


def test_more_methods_than_markers():
    """A 21st method used to raise IndexError on the hard-coded 20-marker list."""
    assert len(MARKERS) > 0
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            os.makedirs('results/exp/cifar10/0', exist_ok=True)
            plot_trail_acc(
                dataset_name='cifar10', trial=0,
                methods_results={f'ResNet18_m{i}': {'test_accuracy': [1., 2.]} for i in range(25)},
                n_lab_obs=[100, 200], exp_path='exp', key='test_accuracy',
                plot_png_name='cifar10/0/results.png',
            )
        finally:
            os.chdir(cwd)


def test_count_class_observation_accepts_array_indices():
    """`topk_idx_obs == None` on a numpy array raises 'truth value ... is ambiguous'."""
    classes = ['a', 'b', 'c']
    ds = _FakeDS([0, 1, 2, 1, 1])

    assert count_class_observation(classes, ds) == {'a': 1, 'b': 3, 'c': 1}
    assert count_class_observation(classes, ds, topk_idx_obs=[0, 1]) == {'a': 1, 'b': 1, 'c': 0}
    assert count_class_observation(classes, ds, topk_idx_obs=np.array([0, 1])) == {'a': 1, 'b': 1, 'c': 0}
    assert count_class_observation(classes, ds, topk_idx_obs=torch.tensor([0, 1])) == {'a': 1, 'b': 1, 'c': 0}


def test_count_class_observation_flags_out_of_range_label():
    try:
        count_class_observation(['a', 'b'], _FakeDS([0, 5]))
    except AssertionError:
        return
    raise AssertionError('a label outside the class list should be reported, not silently counted')


def test_write_csv_roundtrips_without_blank_rows():
    """Without `newline=''` the csv module emits blank lines that pandas reads back as NaN."""
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            head = ['method', 'iter', 'lab_obs', 'test_accuracy']
            for i in range(3):
                write_csv('results.csv', 'exp', 'cifar10', head,
                          ['ResNet18_Random', 0, 100 * (i + 1), 0.5 + i / 10])

            df = pd.read_csv('results/exp/cifar10/results.csv')
            assert list(df.columns) == head
            assert len(df) == 3 and not df.isna().any().any()
        finally:
            os.chdir(cwd)


def test_write_csv_creates_missing_directory():
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            write_csv('r.csv', 'brand_new', 'cifar10', ['a'], [1])
            assert os.path.exists('results/brand_new/cifar10/r.csv')
        finally:
            os.chdir(cwd)


def test_mean_std_has_no_nan_with_a_single_trial():
    """One trial -> pandas std is NaN -> NaN CI, invisible band and NaNs in the saved CSV."""
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            os.makedirs('results/exp/cifar10', exist_ok=True)
            pd.DataFrame({
                'method': ['ResNet18_Random'] * 3 + ['ResNet18_GTG_off'] * 3,
                'iter': [0] * 6,
                'lab_obs': [100, 200, 300] * 2,
                'test_accuracy': [50., 55., 60., 52., 58., 63.],
            }).to_csv('results/exp/cifar10/results.csv', index=False)

            plot_res_std_mean(exp_path='exp', dataset_name='cifar10')

            out = pd.read_csv('results/exp/cifar10/mean_std_clf_results.csv')
            assert not out.isna().any().any(), out
            assert (out['ci_lower'] == out['mean']).all()  # zero-width CI for n = 1
        finally:
            os.chdir(cwd)


def test_mean_std_ci_uses_student_t():
    """z = 1.96 understated the 95% CI over 5 trials by ~30%; the half-width must use t(0.975, n-1)."""
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            os.makedirs('results/exp/cifar10', exist_ok=True)
            accuracies = [50., 52., 55., 57., 61.]
            pd.DataFrame({
                'method': ['ResNet_Random'] * 5, 'iter': list(range(5)),
                'lab_obs': [100] * 5, 'test_accuracy': accuracies,
            }).to_csv('results/exp/cifar10/results.csv', index=False)

            plot_res_std_mean(exp_path='exp', dataset_name='cifar10')

            out = pd.read_csv('results/exp/cifar10/mean_std_clf_results.csv')
            expected = stats.t.ppf(0.975, 4) * np.std(accuracies, ddof=1) / np.sqrt(5)
            assert np.isclose(out['ci_upper'].iloc[0] - out['mean'].iloc[0], expected), out
        finally:
            os.chdir(cwd)


def test_save_figure_survives_a_render_failure():
    """A broken font stack raised inside savefig after every AL cycle and killed the whole run."""
    def broken_savefig(self, *args, **kwargs):
        raise RuntimeError('FT_Render_Glyph failed with error 0x62: raster overflow')

    original_savefig = matplotlib.figure.Figure.savefig
    matplotlib.figure.Figure.savefig = broken_savefig
    plt.close('all')
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                keys = ['train_accuracy', 'train_loss', 'train_loss_ce', 'train_pred_loss']
                save_train_val_curves(
                    list_dict_keys=keys, results_info={k: [1., 2.] for k in keys},
                    strategy_name='ResNet_Random', exp_path='exp', dataset_name='cifar10',
                    al_iter=1, cicle_iter=0,
                )  # must not raise
                assert len(plt.get_fignums()) == 0, 'the figure was not closed after the failure'
            finally:
                os.chdir(cwd)
    finally:
        matplotlib.figure.Figure.savefig = original_savefig


def test_purge_rows_removes_only_the_target_trial():
    """Resume discards a partial trial; every other row must survive untouched."""
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            head = ['method', 'iter', 'lab_obs', 'test_accuracy']
            rows = [['ResNet_Random', 0, 1000, 0.5], ['ResNet_Random', 1, 1000, 0.25],
                    ['ResNet_Entropy', 1, 1000, 0.125], ['ResNet_Random', 1, 2000, 0.375]]
            for row in rows:
                write_csv(filename='results.csv', exp_path='exp', dataset_name='cifar10', head=head, values=row)

            n_removed = purge_rows('results.csv', 'exp', 'cifar10', 'ResNet_Random', 1)

            assert n_removed == 2
            assert read_rows('results.csv', 'exp', 'cifar10') == [
                ['ResNet_Random', '0', '1000', '0.5'], ['ResNet_Entropy', '1', '1000', '0.125']]
            assert purge_rows('missing.csv', 'exp', 'cifar10', 'ResNet_Random', 1) == 0
        finally:
            os.chdir(cwd)


def test_mean_std_handles_a_non_accuracy_metric():
    """Hard-coding 'test_accuracy' raised KeyError at the end of a sweep on any other task."""
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            os.makedirs('results/exp/voc', exist_ok=True)
            pd.DataFrame({
                'method': ['ResNet18_Random'] * 2,
                'iter': [0, 0],
                'lab_obs': [100, 200],
                'test_map': [0.4, 0.5],
            }).to_csv('results/exp/voc/results.csv', index=False)

            plot_res_std_mean(exp_path='exp', dataset_name='voc')  # must not raise
            assert os.path.exists('results/exp/voc/mean_std_results.png')
        finally:
            os.chdir(cwd)


def test_gtg_entropy_plot_survives_early_convergence():
    """GTG may stop before `gtg_max_iter`; `np.arange(max_x)` then mismatched the data."""
    with tempfile.TemporaryDirectory() as tmp:
        n_obs, actual_iters, declared_max = 12, 7, 20
        tensor = torch.rand(n_obs, actual_iters)

        plot_gtg_entropy_tensor(
            tensor=tensor, topk=[0, 3, 5], lab_unlabels=[i % 4 for i in range(n_obs)],
            classes=['a', 'b', 'c', 'd'], path=tmp, iter=1, max_x=declared_max, dir='history',
        )
        assert os.path.exists(os.path.join(tmp, 'gtg_entropies_plots', 'history', '1.png'))


def test_gtg_entropy_plot_with_many_classes():
    """The >10-class branch indexes the palette by label; it must not fall off the end."""
    with tempfile.TemporaryDirectory() as tmp:
        n_classes = 120
        tensor = torch.rand(30, 5)
        plot_gtg_entropy_tensor(
            tensor=tensor, topk=[1, 2], lab_unlabels=[i % n_classes for i in range(30)],
            classes=[f'c{i}' for i in range(n_classes)], path=tmp, iter=0, max_x=5, dir='derivatives',
        )
        assert os.path.exists(os.path.join(tmp, 'gtg_entropies_plots', 'derivatives', '0.png'))


def test_entropy_is_finite_on_a_degenerate_distribution():
    p = torch.tensor([[1., 0., 0., 0.]])
    e = entropy(p, dim=1)
    assert torch.isfinite(e).all() and e.item() < 1e-5


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f'  PASS  {t.__name__}')
        except Exception as e:
            failed += 1
            print(f'  FAIL  {t.__name__}: {type(e).__name__}: {e}')
    print(f'\n{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)

"""Regression tests for the pipeline fixes that protect the reported numbers.

Covers the training/test worker, TiDAL's shared moving average, trial-level resume in `main.py`,
the sweep state in `run_sweep.py` and the refusal checks in `docs/paper/make_results.py`. Fast:
no training and no real dataset.

    python app/tests/test_pipeline_fixes.py
"""

import os
import sys
import tempfile

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
REPO_DIR = os.path.join(APP_DIR, '..')
sys.path.insert(0, APP_DIR)
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, 'docs', 'paper'))

import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

import config
from config import al_params
from utils import accuracy_score, read_rows, write_csv


class _FakeTorchvisionDS:
    """Stands in for a torchvision dataset class in `config.cls_datasets` (no download)."""

    def __init__(self, root, train, download, transform):
        self.n = 64

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.zeros(3, 4, 4), i % 3


def test_tidal_moving_prob_is_visible_to_forked_workers():
    """The Worker updates moving_prob in the main process; forked workers (Linux default) read a
    private copy unless the tensor lives in shared memory, so TiDAL's target stayed at zero."""
    from dataset import Cls_Dataset

    config.cls_datasets['_fake'] = {'method': _FakeTorchvisionDS, 'transforms': {'train': None, 'test': None}}
    try:
        ds = Cls_Dataset('_fake', bool_train=True, bool_transform=True, n_classes=3)
        assert ds.moving_prob.is_shared(), 'moving_prob is not in shared memory'

        if 'fork' not in mp.get_all_start_methods(): return
        dl = DataLoader(ds, sampler=SubsetRandomSampler(list(range(len(ds)))), batch_size=16,
                        num_workers=2, persistent_workers=True, multiprocessing_context='fork')
        seen_non_zero = []
        for epoch in range(2):
            n_non_zero = 0
            for idxs, _, _, moving_prob in dl:
                n_non_zero += int((moving_prob.sum(dim=1) > 0).sum())
                dl.dataset.moving_prob[idxs.tolist(), :] = float(epoch + 1)
            seen_non_zero.append(n_non_zero)
        assert seen_non_zero == [0, len(ds)], f'workers did not see the update: {seen_non_zero}'
    finally:
        del config.cls_datasets['_fake']


def test_worker_test_accuracy_weights_the_last_batch():
    """Averaging per-batch accuracies over-weighted the short last batch (0.667 instead of 0.8 here)."""
    from train_evaluate.Worker import Cls_TrainWorker

    class _Model:
        def backbone(self, images): return images, None    # the "images" are already logits
        def eval(self): return self

    labels = torch.zeros(10, dtype=torch.long)
    logits = torch.tensor([[1., 0.]] * 8 + [[0., 1.]] * 2)   # batches of 4, 4, 2 -> 1, 1, 0 accuracy

    worker = Cls_TrainWorker.__new__(Cls_TrainWorker)
    worker._Cls_TrainWorker__load_checkpoint = lambda filename: None
    worker.check_best_path, worker.model, worker.score_fn = 'unused', _Model(), accuracy_score
    worker.world_size, worker.device = 1, torch.device('cpu')
    worker.test_dl = DataLoader(TensorDataset(torch.arange(10), logits, labels), batch_size=4, shuffle=False)

    assert np.isclose(worker.test().item(), 0.8), worker.test().item()


class _FakeStrategy:
    def __init__(self, name):
        self.strategy_name, self.n_runs = name, 0

    def run(self):
        self.n_runs += 1
        return {'test_accuracy': [-1.]}


def _expected_lab_obs():
    return [al_params['init_lab_obs'] + i * al_params['n_top_k_obs'] for i in range(al_params['al_iters'])]


def test_resume_skips_complete_trials_and_reruns_partial_ones():
    """A crashed method used to restart at trial 0 and append its finished trials a second time."""
    import main as app_main

    lab_obs = _expected_lab_obs()
    results_head = ['method', 'iter', 'lab_obs', 'test_accuracy']
    counts_head = ['method', 'iter', 'lab_obs', 'a', 'b']

    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            for position, budget in reversed(list(enumerate(lab_obs))):     # written out of order
                write_csv('results.csv', 'exp', 'cifar10', results_head, ['S_done', 0, budget, position / 10])
            for budget in lab_obs[:2]:
                write_csv('results.csv', 'exp', 'cifar10', results_head, ['S_partial', 0, budget, 0.5])
            for filename in ('count_classes.csv', '_count_classes.csv'):
                write_csv(filename, 'exp', 'cifar10', counts_head, ['S_partial', 0, lab_obs[0], 1, 2])
                write_csv(filename, 'exp', 'cifar10', counts_head, ['S_partial', 1, lab_obs[0], 3, 4])

            done = _FakeStrategy('S_done')
            out = app_main.run_strategy(done, 0, 'exp', 'cifar10', 'resume', lab_obs)
            assert done.n_runs == 0, 'a complete trial was re-run'
            assert out == {'test_accuracy': [position / 10 for position in range(len(lab_obs))]}, out

            partial = _FakeStrategy('S_partial')
            app_main.run_strategy(partial, 0, 'exp', 'cifar10', 'resume', lab_obs)
            assert partial.n_runs == 1, 'a partial trial was not re-run'
            assert not [r for r in read_rows('results.csv', 'exp', 'cifar10') if r[0] == 'S_partial']
            for filename in ('count_classes.csv', '_count_classes.csv'):
                assert read_rows(filename, 'exp', 'cifar10') == [['S_partial', '1', str(lab_obs[0]), '3', '4']]

            app_main.run_strategy(done, 0, 'exp', 'cifar10', 'overwrite', lab_obs)
            assert done.n_runs == 1 and not read_rows('results.csv', 'exp', 'cifar10'), 'overwrite kept old rows'

            write_csv('results.csv', 'exp', 'cifar10', results_head, ['S_done', 0, lab_obs[0], 0.1])
            app_main.run_strategy(done, 0, 'exp', 'cifar10', 'append', lab_obs)
            assert done.n_runs == 2 and len(read_rows('results.csv', 'exp', 'cifar10')) == 1
        finally:
            os.chdir(cwd)


def _parse_main_args(argv):
    import main as app_main
    old_argv = sys.argv
    sys.argv = ['main.py'] + argv
    try:
        return app_main.get_args()
    finally:
        sys.argv = old_argv


def test_online_gtg_refuses_several_affinity_configs():
    """All online strategies share one GTGModule: with 2 -am_ts values both trained with the last."""
    try:
        _parse_main_args(['-m', 'lsmlps_gtg', '-ds', 'cifar10', '-am_ts', 'mean', 'none'])
    except SystemExit as exit_:
        assert exit_.code == 2
    else:
        raise AssertionError('several -am_ts values were accepted for an online GTG method')

    assert _parse_main_args(['-m', 'lsmlps_gtg', '-ds', 'cifar10']).affinity_matrix_threshold_strategies == ['mean']
    assert _parse_main_args(['-m', 'gtg', '-ds', 'cifar10', '-am_ts', 'mean', 'none'])   # offline keeps its own config


def _run_sweep(argv, return_code=0):
    """Run run_sweep.main() in-process; return (exit code, main.py command lines it launched)."""
    import run_sweep
    launched = []

    def fake_call(cmd, **kwargs):
        launched.append(cmd)
        return return_code

    old_call, old_argv = run_sweep.subprocess.call, sys.argv
    run_sweep.subprocess.call, sys.argv = fake_call, ['run_sweep.py'] + argv
    try:
        run_sweep.main()
        code = 0
    except SystemExit as exit_:
        code = exit_.code
    finally:
        run_sweep.subprocess.call, sys.argv = old_call, old_argv
    return code, launched


def _existing_flag(cmd):
    return cmd[cmd.index('--existing') + 1]


def test_run_sweep_resumes_and_refuses_mixed_setups():
    """A 1-trial timing run marked a method done, so the 5-trial sweep skipped it; a crash appended
    duplicates on re-run; and a different --extra silently mixed configurations in one -o."""
    import run_sweep

    with tempfile.TemporaryDirectory() as tmp:
        old_here, run_sweep.HERE = run_sweep.HERE, tmp
        try:
            base = ['-o', 'exp', '-ds', 'cifar10', '-m', 'random']

            code, launched = _run_sweep(base + ['--trials', '2'], return_code=1)          # crash
            assert code == 1 and _existing_flag(launched[0]) == 'resume'
            assert run_sweep.read_state(run_sweep.state_path('exp', 'cifar10', 'random'))['status'] == 'running'

            code, launched = _run_sweep(base + ['--trials', '2'])                          # resume
            assert code == 0 and len(launched) == 1 and _existing_flag(launched[0]) == 'resume'

            code, launched = _run_sweep(base + ['--trials', '2'])                          # done: skip
            assert code == 0 and launched == []

            code, launched = _run_sweep(base + ['--trials', '3'])                          # more trials
            assert code == 0 and len(launched) == 1 and _existing_flag(launched[0]) == 'resume'

            for argv in (base + ['--trials', '2'], base + ['--trials', '3', '--extra', '-am_r', '0.5']):
                code, launched = _run_sweep(argv)                                          # refused
                assert code == 2 and launched == [], argv

            code, launched = _run_sweep(base + ['--trials', '3', '--force'])
            assert code == 0 and _existing_flag(launched[0]) == 'overwrite'

            open(run_sweep.legacy_marker('exp', 'cifar10', 'entropy'), 'w').close()
            code, launched = _run_sweep(['-o', 'exp', '-ds', 'cifar10', '-m', 'entropy'])
            assert code == 2 and launched == []
        finally:
            run_sweep.HERE = old_here


def _results_frame(rows):
    import make_results
    df = pd.DataFrame(rows, columns=['method', 'iter', 'lab_obs', 'test_accuracy'])
    df['short'] = df['method'].map(make_results._short_name)
    return df


def _refused(df, trials):
    import make_results
    try:
        make_results.validate(df, trials)
    except SystemExit:
        return True
    return False


def test_make_results_refuses_rows_that_would_distort_the_ci():
    """Duplicates, missing trials and pooled strategies each inflated `count` and narrowed the CI."""
    import make_results

    offline, offline_ll = 'ResNet_GTG_off_corr_mixed_ts-mean_es-mean', 'ResNet_LossNet_1_GTG_off_corr_mixed_ts-mean_es-mean'
    clean = [[method, trial, budget, acc]
             for method in ('ResNet_Random', offline, offline_ll)
             for trial, acc in ((0, 0.50), (1, 0.54))
             for budget in (1000, 2000)]
    assert not _refused(_results_frame(clean), trials=2), 'gtg and ll_gtg must be separate table rows'

    assert _refused(_results_frame(clean + [clean[0]]), trials=2), 'duplicated row accepted'
    assert _refused(_results_frame(clean), trials=3), 'missing trials accepted'
    pooled = clean + [['ResNet_GTG_off_corr_mixed_es-mean', trial, budget, 0.5]
                      for trial in (0, 1) for budget in (1000, 2000)]
    assert _refused(_results_frame(pooled), trials=2), 'two configurations pooled into one row'

    g = make_results.aggregate(_results_frame(clean))
    row = g[(g['short'] == 'Random') & (g['lab_obs'] == 1000)].iloc[0]
    expected = stats.t.ppf(0.975, 1) * np.std([50., 54.], ddof=1) / np.sqrt(2)
    assert np.isclose(row['ci'], expected), (row['ci'], expected)


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

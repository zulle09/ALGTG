"""
Regenerates bait_golden.json by running the REFERENCE BAIT code in isolation on a small synthetic fixture.
A development tool: nothing under app/ imports the reference code, and tests/test_bait.py only reads the JSON
written here.

  - dal-toolbox, dal_toolbox/active_learning/strategies/bait.py (BaitSampling), used as a numerical oracle

    python app/tests/golden/make_bait_golden.py --dal-toolbox <dal-toolbox clone>

The reference is run with fisher_batch_size larger than the data, so that its Fisher matrices are the global
means (with its default of 32 it adds one mean per batch, which scales them by the number of batches). It
must agree with tests/golden/dense_bait.py, an implementation with explicit matrices and explicit inverses,
and the fixture is searched until no step of the greedy has a near-tie, so float32 rounding in the reference
cannot change a pick. Needs numpy and torch.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import warnings

import numpy as np
import torch

import reference_stubs as stubs
from dense_bait import dense_bait

N_CLASSES, DIM, N_TOTAL = 4, 6, 52
# (name, fisher, number of labelled points, budget, lambda)
SCENARIOS = [
    ('ce_with_labelled', 'ce', 12, 5, 0.5),
    ('ce_without_labelled', 'ce', 0, 5, 1.0),
    ('bce_with_labelled', 'bce', 12, 5, 0.5),
    ('ce_with_the_cifar10_lambda', 'ce', 20, 8, 0.01),
]
MIN_GAP = 1e-5


def make_fixture(seed):
    """Non-negative features (as after a ReLU) and logits of mixed confidence; one candidate is saturated."""
    rng = np.random.default_rng(seed)
    features = np.round(np.abs(rng.normal(size=(N_TOTAL, DIM))), 6)
    logits = np.round(rng.normal(scale=1.5, size=(N_TOTAL, N_CLASSES)), 6)
    logits[-1] = [30.0, 0.0, 0.0, 0.0]
    return features, logits


def load_dal(path):
    stubs.install_stub('dalstub', __path__=[])
    class Query:   # dal_toolbox's base class, which bait.py imports relatively
        def __init__(self, random_seed=None):
            self.random_seed = random_seed
    stubs.install_stub('dalstub.query', Query=Query)
    return stubs.load_module('dalstub.bait', path)


def run_dal(module, features, logits, labelled, pool, kind, budget, lam):
    class Model:
        def get_model_outputs(self, loader, output_types, device='cpu'):
            ids = np.asarray(loader, dtype=int)
            return {'features': torch.tensor(features[ids], dtype=torch.float32), 'logits': torch.tensor(logits[ids], dtype=torch.float32)}

    class DataModule:
        def unlabeled_dataloader(self, subset_size=None):
            return list(pool), list(pool)

        def labeled_dataloader(self):
            return list(labelled), list(labelled)

    grad_likelihood = {'ce': 'cross_entropy', 'bce': 'binary_cross_entropy'}[kind]
    strategy = module.BaitSampling(lmb=lam, grad_likelihood=grad_likelihood, fisher_batch_size=10 ** 6)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        picks = strategy.query(model=Model(), al_datamodule=DataModule(), acq_size=budget)
    return [int(pick) for pick in picks]


def sha256(path):
    with open(path, 'rb') as file:
        return hashlib.sha256(file.read()).hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dal-toolbox', required=True, help='root of a dal-toolbox clone')
    parser.add_argument('--out', default=os.path.join(here, 'bait_golden.json'))
    args = parser.parse_args()

    dal_file = os.path.join(args.dal_toolbox, 'dal_toolbox', 'active_learning', 'strategies', 'bait.py')
    dal = load_dal(dal_file)

    for seed in range(300):
        features, logits = make_fixture(seed)
        rng = np.random.default_rng(seed + 1000)
        scenarios, smallest_gap, agree = [], np.inf, True
        for name, kind, n_labelled, budget, lam in SCENARIOS:
            labelled = sorted(int(i) for i in rng.choice(N_TOTAL - 1, n_labelled, replace=False))
            pool = [int(i) for i in rng.permutation(np.setdiff1d(np.arange(N_TOTAL), labelled))]
            dense, gaps = dense_bait(logits[labelled], features[labelled], logits[pool], features[pool], budget, lam, kind)
            expected = run_dal(dal, features, logits, labelled, pool, kind, budget, lam)
            smallest_gap = min(smallest_gap, min(gaps))
            agree &= [pool[i] for i in dense] == expected
            scenarios.append(dict(name=name, fisher=kind, lambda_reg=lam, labelled=labelled, pool=pool, budget=budget, expected=expected))
        print(f'fixture seed {seed}: smallest relative gap between the best and the second best step {smallest_gap:.1e}, reference == dense: {agree}')
        if smallest_gap > MIN_GAP and agree:
            break
    else:
        sys.exit('no fixture without near-ties in 300 seeds')

    golden = dict(
        meta=dict(
            generated_by='app/tests/golden/make_bait_golden.py',
            seed=seed,
            dal_toolbox=dict(repo='https://github.com/dhuseljic/dal-toolbox',
                             commit=subprocess.run(['git', '-C', args.dal_toolbox, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
                             sha256=sha256(dal_file), file='dal_toolbox/active_learning/strategies/bait.py', licence='BSD-3-Clause'),
            note='Reference run with fisher_batch_size >= n (global means). Used as a numerical oracle only; the port is '
                 'written from the paper, see strategies/competitors/BAIT.py.',
        ),
        features=features.tolist(), logits=logits.tolist(), scenarios=scenarios,
    )
    with open(args.out, 'w') as file:
        json.dump(golden, file)
    for scenario in scenarios:
        print(f'{scenario["name"]:32s} {scenario["fisher"]:3s} L={len(scenario["labelled"]):2d} B={scenario["budget"]}  {scenario["expected"]}')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

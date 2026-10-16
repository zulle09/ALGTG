"""
Regenerates probcover_golden.json by running the REFERENCE ProbCover code in isolation on a small
synthetic fixture. A development tool: nothing under app/ imports the reference code, and
tests/test_probcover.py only reads the JSON written here.

  - dal:     dal-toolbox, dal_toolbox/active_learning/strategies/prob_cover.py   (BSD-3-Clause)
  - authors: the authors' implementation, as shipped in the uherding repository, deep-al/pycls/al/prob_cover.py

    python app/tests/golden/make_probcover_golden.py --dal-toolbox <dal-toolbox clone> --uherding <uherding clone>

Both references build their graph over `range(n // batch_size)` batches, which silently drops the out-edges
of the last `n mod batch_size` points, and both repeat index 0 once nothing is left to cover. The generator
runs them with one batch covering everything, and records where they run out; the port does not share either
behaviour. The two references must agree with each other and with an independent brute-force greedy before
anything is written. The radii are chosen away from every pairwise distance, so float32 rounding cannot flip
an edge. Needs numpy, torch and pandas.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import warnings

import numpy as np
import torch

import reference_stubs as stubs


N_POINTS, DIM, N_CLUSTERS = 90, 6, 5
# (name, radius, number of labelled points, budget, whether coverage must run out before the budget does)
SCENARIOS = [
    ('moderate_radius_with_labelled', 'moderate', 12, 6, False),
    ('moderate_radius_without_labelled', 'moderate', 0, 8, False),
    ('many_labelled_points', 'moderate', 24, 5, False),
    ('coverage_exhausted_before_the_budget', 'large', 4, 10, True),
]


def make_fixture():
    """Clustered unit vectors, then two radii in the widest gaps between pairwise distances."""
    rng = np.random.default_rng(3)
    centres = rng.normal(size=(N_CLUSTERS, DIM))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    points = centres[rng.integers(0, N_CLUSTERS, size=N_POINTS)] + 0.15 * rng.normal(size=(N_POINTS, DIM))
    points = np.round(points / np.linalg.norm(points, axis=1, keepdims=True), 7).astype(np.float32)

    distances = np.sort(np.linalg.norm(points[:, None].astype(np.float64) - points[None], axis=-1)[np.triu_indices(N_POINTS, 1)])
    gaps = np.diff(distances)

    def widest_gap_between(low, high):
        inside = np.flatnonzero((distances[:-1] >= low) & (distances[1:] <= high))
        best = inside[gaps[inside].argmax()]
        return float(round((distances[best] + distances[best + 1]) / 2, 6)), float(gaps[best] / 2)

    radii = {'moderate': widest_gap_between(0.30, 0.42), 'large': widest_gap_between(0.45, 0.55)}
    for name, (radius, margin) in radii.items():
        assert margin > 5e-4, f'{name}: the nearest pairwise distance is only {margin:.1e} from the radius'
    return points, {name: radius for name, (radius, _) in radii.items()}


def brute_force_greedy(points, labelled, pool, delta, budget):
    """Independent re-implementation with plain loops in float64: (picks, number of picks with a positive gain)."""
    points = points.astype(np.float64)
    dist = lambda a, b: np.linalg.norm(points[a] - points[b])
    covered = {y for y in pool if any(dist(x, y) < delta for x in labelled)}
    balls = {c: {y for y in pool if y == c or dist(c, y) < delta} for c in pool}
    picks = []
    while len(picks) < budget:
        gain = {c: len(balls[c] - covered) for c in pool if c not in picks}
        best = max(gain.values())
        if best == 0:
            break
        chosen = next(c for c in pool if c not in picks and gain[c] == best)   # lowest pool position wins ties
        covered |= balls[chosen]
        picks.append(chosen)
    return picks


def run_dal(module, points, labelled, pool, delta, budget):
    class Model:
        def get_representations(self, loader):
            return torch.tensor(points[np.asarray(loader, dtype=int)])

    class DataModule:
        def unlabeled_dataloader(self, subset_size=None):
            return list(pool), list(pool)

        def labeled_dataloader(self):
            return list(labelled), list(labelled)

    n_nodes = len(labelled) + len(pool)
    module.construct_graph.__defaults__ = (n_nodes,)   # one batch: keep the out-edges of the last rows
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        picks = module.ProbCover(delta=delta).query(model=Model(), al_datamodule=DataModule(), acq_size=budget)
    return [int(pick) for pick in picks]


def run_authors(module, points, labelled, pool, delta, budget):
    class Config(dict):
        __getattr__ = dict.__getitem__

    config = Config(DATASET=Config(NAME='FIXTURE'), RNG_SEED=0, ACTIVE_LEARNING=Config(UNC_FEATURE='simclr'))
    dataset = type('Dataset', (), {'features': points})()   # the loader below L2-normalises, as the real one does

    original = module.ProbCover.construct_graph
    module.ProbCover.construct_graph = lambda self, batch_size=500: original(self, batch_size=len(self.rel_features))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        strategy = module.ProbCover(config, np.asarray(labelled, dtype=int), np.asarray(pool, dtype=int), budget, delta, dataset)
        selected, _ = strategy.select_samples()
    module.ProbCover.construct_graph = original
    return [int(pick) for pick in selected]


def sha256(path):
    with open(path, 'rb') as file:
        return hashlib.sha256(file.read()).hexdigest()


def git_head(repository):
    return subprocess.run(['git', '-C', repository, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dal-toolbox', required=True, help='root of a dal-toolbox clone')
    parser.add_argument('--uherding', required=True, help='root of a uherding clone')
    parser.add_argument('--out', default=os.path.join(here, 'probcover_golden.json'))
    args = parser.parse_args()

    dal_file = os.path.join(args.dal_toolbox, 'dal_toolbox', 'active_learning', 'strategies', 'prob_cover.py')
    authors_file = os.path.join(args.uherding, 'deep-al', 'pycls', 'al', 'prob_cover.py')

    stubs.install_dal_toolbox_stubs()
    dal = stubs.load_module('reference_dal_prob_cover', dal_file)
    pycls_utils = stubs.install_pycls_stubs()
    pycls_utils.load_features = lambda *a, dataset=None, **k: dataset.features / np.linalg.norm(dataset.features, axis=1, keepdims=True)
    authors = stubs.load_module('reference_authors_prob_cover', authors_file)

    points, radii = make_fixture()
    rng = np.random.default_rng(1)
    scenarios = []
    with stubs.cuda_as_identity():
        for name, radius_name, n_labelled, budget, must_exhaust in SCENARIOS:
            delta = radii[radius_name]
            labelled = sorted(int(i) for i in rng.choice(N_POINTS, n_labelled, replace=False))
            pool = [int(i) for i in rng.permutation(np.setdiff1d(np.arange(N_POINTS), labelled))]

            expected_dal = run_dal(dal, points, labelled, pool, delta, budget)
            expected_authors = run_authors(authors, points, labelled, pool, delta, budget)
            greedy = brute_force_greedy(points, labelled, pool, delta, budget)
            n_deterministic = len(greedy)

            assert expected_dal == expected_authors, f'{name}: the two references disagree\n{expected_dal}\n{expected_authors}'
            assert expected_dal[:n_deterministic] == greedy, f'{name}: reference {expected_dal} != brute force {greedy}'
            reference_repeats = len(set(expected_dal)) < len(expected_dal) or bool(set(expected_dal) & set(labelled))
            assert reference_repeats == (n_deterministic < budget), name
            assert (n_deterministic < budget) == must_exhaust, f'{name}: {n_deterministic} greedy picks for a budget of {budget}'
            scenarios.append(dict(
                name=name, delta=delta, labelled=labelled, pool=pool, budget=budget,
                n_deterministic=n_deterministic, reference_repeats_or_picks_labelled=reference_repeats, expected=expected_dal,
            ))
            print(f'{name:42s} delta {delta:.4f} budget {budget:2d}  greedy picks {n_deterministic:2d}  reference {expected_dal}')

    golden = dict(
        meta=dict(
            generated_by='app/tests/golden/make_probcover_golden.py',
            dal_toolbox=dict(repo='https://github.com/dhuseljic/dal-toolbox', commit=git_head(args.dal_toolbox), sha256=sha256(dal_file),
                             file='dal_toolbox/active_learning/strategies/prob_cover.py', licence='BSD-3-Clause'),
            authors_copy=dict(repo='https://github.com/BorealisAI/uherding', commit=git_head(args.uherding), sha256=sha256(authors_file),
                              file='deep-al/pycls/al/prob_cover.py', licence='header: MIT, from github.com/avihu111/TypiClust'),
            note='Both references run with one batch (no dropped rows). Only the first n_deterministic picks are compared: '
                 'after that the references repeat index 0 and the port draws distinct random candidates.',
        ),
        features=points.tolist(),
        scenarios=scenarios,
    )
    with open(args.out, 'w') as file:
        json.dump(golden, file)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

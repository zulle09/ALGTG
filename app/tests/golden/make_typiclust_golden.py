"""
Regenerates typiclust_golden.json by running the REFERENCE TypiClust code in isolation on a small
synthetic fixture. A development tool: nothing under app/ imports the reference code, and
tests/test_typiclust.py only reads the JSON written here.

Both references get the same injected clustering (k-means is random and differs between them), so
what is compared is the selection logic:
  - dal:     dal-toolbox, dal_toolbox/active_learning/strategies/typiclust.py   (BSD-3-Clause)
  - authors: https://github.com/avihu111/TypiClust, deep-al/pycls/al/typiclust.py   (MIT)

    python app/tests/golden/make_typiclust_golden.py \\
        --dal-toolbox  <clone of github.com/dhuseljic/dal-toolbox> \\
        --typiclust-original <typiclust.py of github.com/avihu111/TypiClust> [--typiclust-commit <sha>]

Needs numpy, torch, scikit-learn and pandas. The heavy imports of the references (lightning, faiss,
the repositories' own packages) are replaced by stubs below.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import types
import warnings

import numpy as np
import torch


CLUSTER_SIZES = [47, 40, 35, 30, 25, 20, 1, 2]     # six blobs, a lone outlier and a pair
DIM = 8
MAX_CLUSTERS = 8

# (name, rule, number of labelled points, budget)
SCENARIOS = [
    ('dal_typical_only', 'dal', 12, 3),
    ('dal_every_uncovered_cluster_once', 'dal', 12, 5),
    ('dal_with_a_random_tail', 'dal', 12, 9),
    ('dal_without_labelled_data', 'dal', 0, 10),
    ('authors_first_round', 'authors', 12, 3),
    ('authors_cycles_through_the_clusters', 'authors', 12, 20),
    ('authors_without_labelled_data', 'authors', 0, 10),
]


def make_fixture():
    """Well-separated clusters whose density falls off with the distance to the centre, so the most
    typical point is never a near-tie between two numerically different distance routines."""
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(len(CLUSTER_SIZES), DIM)) * 20
    points, cluster_ids = [], []
    for cluster, (centre, size) in enumerate(zip(centres, CLUSTER_SIZES)):
        directions = rng.normal(size=(size, DIM))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = 0.05 * (1 + np.arange(size)) ** 1.5
        points.append(centre + radii[:, None] * directions)
        cluster_ids.append(np.full(size, cluster))
    features = np.round(np.concatenate(points), 6).astype(np.float32)
    return features, np.concatenate(cluster_ids)


def install_stub(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dal_reference(path):
    class Query:   # what dal_toolbox's Query base class provides: self.rng from setup_rng(seed)
        def __init__(self, random_seed=None):
            self.random_seed = random_seed
            self.rng = np.random.RandomState(random_seed) if random_seed is not None else np.random.mtrand._rand

    install_stub('dal_toolbox')
    install_stub('dal_toolbox.active_learning', ActiveLearningDataModule=object)
    install_stub('dal_toolbox.active_learning.strategies', Query=Query)
    install_stub('dal_toolbox.models')
    install_stub('dal_toolbox.models.base', BaseModule=object)
    return load_module('reference_dal_typiclust', path)


def load_authors_reference(path):
    class IndexFlatL2:   # exact squared-L2 search, what faiss.IndexFlatL2 computes
        def __init__(self, dim):
            self.data = None

        def add(self, data):
            self.data = data.astype(np.float64)

        def search(self, queries, k):
            squared = ((queries.astype(np.float64)[:, None, :] - self.data[None, :, :]) ** 2).sum(-1)
            nearest = np.argsort(squared, axis=1, kind='stable')[:, :k]
            return np.take_along_axis(squared, nearest, axis=1).astype(np.float32), nearest

    install_stub('faiss', IndexFlatL2=IndexFlatL2, index_cpu_to_all_gpus=lambda index: index)
    install_stub('pycls')
    datasets = install_stub('pycls.datasets')
    datasets.utils = install_stub('pycls.datasets.utils', load_features=None)
    sys.modules['pycls'].datasets = datasets
    return load_module('reference_authors_typiclust', path)


def run_dal(module, features, cluster_ids, labelled, pool, budget):
    expected_clusters = min(len(labelled) + budget, MAX_CLUSTERS)
    injected = cluster_ids[np.concatenate([labelled, pool]).astype(int)]

    def injected_kmeans(_features, num_clusters):
        assert num_clusters == expected_clusters, (num_clusters, expected_clusters)
        return injected

    class Model:
        def get_model_outputs(self, loader, output_types, device='cpu'):
            return {'features': torch.tensor(features[np.asarray(loader, dtype=int)])}

    class DataModule:
        def unlabeled_dataloader(self, subset_size=None):
            return list(pool), list(pool)

        def labeled_dataloader(self):
            return list(labelled), list(labelled)

    module.kmeans = injected_kmeans
    module.TypiClust.MAX_NUM_CLUSTERS = MAX_CLUSTERS
    picks = module.TypiClust(random_seed=0).query(model=Model(), al_datamodule=DataModule(), acq_size=budget)
    return [int(pick) for pick in picks]


def run_authors(module, features, cluster_ids, labelled, pool, budget):
    expected_clusters = min(len(labelled) + budget, MAX_CLUSTERS)

    def injected_kmeans(_features, num_clusters):
        assert num_clusters == expected_clusters, (num_clusters, expected_clusters)
        return cluster_ids

    module.kmeans = injected_kmeans
    module.ds_utils.load_features = lambda name, seed: features
    module.TypiClust.MAX_NUM_CLUSTERS = MAX_CLUSTERS
    strategy = module.TypiClust(
        {'DATASET': {'NAME': 'FIXTURE'}, 'RNG_SEED': 0},
        np.asarray(labelled, dtype=int), np.asarray(pool, dtype=int), budget, is_scan=False
    )
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        selected, _ = strategy.select_samples()
    return [int(pick) for pick in selected]


def sha256(path):
    with open(path, 'rb') as file:
        return hashlib.sha256(file.read()).hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dal-toolbox', required=True, help='root of a dal-toolbox clone')
    parser.add_argument('--typiclust-original', required=True, help="the authors' deep-al/pycls/al/typiclust.py")
    parser.add_argument('--typiclust-commit', default='not recorded', help="commit of the authors' repository the file comes from")
    parser.add_argument('--out', default=os.path.join(here, 'typiclust_golden.json'))
    args = parser.parse_args()

    dal_file = os.path.join(args.dal_toolbox, 'dal_toolbox', 'active_learning', 'strategies', 'typiclust.py')
    dal_commit = subprocess.run(['git', '-C', args.dal_toolbox, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    references = {'dal': load_dal_reference(dal_file), 'authors': load_authors_reference(args.typiclust_original)}
    runners = {'dal': run_dal, 'authors': run_authors}

    features, cluster_ids = make_fixture()
    rng = np.random.default_rng(1)
    scenarios = []
    for name, rule, n_labelled, budget in SCENARIOS:
        per_cluster = n_labelled // 3
        labelled = sorted(int(i) for cluster in range(3 if n_labelled else 0)
                          for i in rng.choice(np.flatnonzero(cluster_ids == cluster), per_cluster, replace=False))
        pool = [int(i) for i in rng.permutation(np.setdiff1d(np.arange(len(features)), labelled))]
        expected = runners[rule](references[rule], features, cluster_ids, labelled, pool, budget)

        if rule == 'dal':
            covered = set(cluster_ids[labelled].tolist())
            uncovered = [c for c in range(len(CLUSTER_SIZES)) if c not in covered]
            n_deterministic = min(budget, len(uncovered))
        else:
            n_deterministic = budget
        assert len(expected) == len(set(expected)) == budget and not set(expected) & set(labelled), name
        scenarios.append(dict(
            name=name, rule=rule, labelled=labelled, pool=pool, budget=budget, max_clusters=MAX_CLUSTERS,
            n_deterministic=n_deterministic, expected=expected,
        ))
        print(f'{name:40s} budget {budget:2d}  deterministic prefix {n_deterministic:2d}  picks {expected}')

    golden = dict(
        meta=dict(
            generated_by='app/tests/golden/make_typiclust_golden.py',
            dal_toolbox=dict(repo='https://github.com/dhuseljic/dal-toolbox', commit=dal_commit, sha256=sha256(dal_file),
                             file='dal_toolbox/active_learning/strategies/typiclust.py', licence='BSD-3-Clause'),
            typiclust_original=dict(repo='https://github.com/avihu111/TypiClust', commit=args.typiclust_commit,
                                    sha256=sha256(args.typiclust_original), file='deep-al/pycls/al/typiclust.py', licence='MIT'),
            note='Identical clustering injected into both references; the dal rule draws its random tail from '
                 'another generator than the port, so only the first n_deterministic picks are compared.',
        ),
        features=features.tolist(),
        cluster_labels=cluster_ids.tolist(),
        scenarios=scenarios,
    )
    with open(args.out, 'w') as file:
        json.dump(golden, file)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

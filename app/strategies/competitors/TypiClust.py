# TypiClust: G. Hacohen, A. Dekel and D. Weinshall, "Active Learning on a Budget: Opposite Strategies
# Suit High and Low Budgets", ICML 2022 (PMLR 162), https://arxiv.org/abs/2202.02794.
# Official code (linked in the abstract): https://github.com/avihu111/TypiClust
#
# Ported, with changes, from dal-toolbox (https://github.com/dhuseljic/dal-toolbox @ 13ee9ca):
# dal_toolbox/active_learning/strategies/typiclust.py and randomclust.py. dal-toolbox licence:
#
#   BSD 3-Clause License
#
#   Copyright (c) 2020 The dal_toolbox developers.
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   * Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer.
#
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#   * Neither the name of the copyright holder nor the names of its
#     contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The 'authors' selection rule and the typicality estimate come from the authors' code, which
# dal-toolbox itself builds on (MIT licence):
#
#   MIT License
#
#   Copyright 2022 (c) Avihu Dekel and Guy Hacohen
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
#
# Changes: pure functions over arrays plus a thin ActiveLearner wrapper; a local seeded generator
# instead of the global RNG; features of the labelled set and of the candidate pool come from a
# configurable source; guards for a budget larger than the pool, clusters that run out of members and
# degenerate clusterings (see typiclust_select).

import numpy as np
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from torch.utils.data import Subset

from ActiveLearner import ActiveLearner
from competitor_config import competitor_params
from strategies.competitors.common import FeatureSourceMixin, check_selection, cluster_features, derive_rng

from typing import Any, Dict, List, Optional, Tuple

import logging
logger = logging.getLogger(__name__)


SELECTIONS = ('authors', 'dal')


def typicality(features: np.ndarray, k_nn: int) -> np.ndarray:
    """Density of every point: 1 / (mean squared distance to its nearest neighbours + 1e-5).

    The number of neighbours is min(k_nn, n // 2), as in both reference codes, so that a small cluster
    estimates its density from at most half of itself. A lone point is infinitely typical. The paper
    (Eq. 4) writes the plain Euclidean distance; both reference codes use the squared one, as here.
    """
    n_points = len(features)
    if n_points <= 1:
        return np.full(n_points, np.inf)
    n_neighbors = min(k_nn, n_points // 2)
    neighbours = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='sqeuclidean').fit(features)
    distances, _ = neighbours.kneighbors(features)
    # column 0 is the point itself
    return 1.0 / (distances[:, 1:].mean(axis=1) + 1e-5)


def _select_dal(
        features: np.ndarray, labels: np.ndarray, n_labelled: int, budget: int, n_clusters: int,
        k_nn: int, rng: np.random.Generator
    ) -> List[int]:
    """dal-toolbox's rule: the most typical point of the largest cluster that holds no labelled point.

    Once every cluster is covered, the remaining picks are uniform over the unselected candidates. With
    K capped below |L| + B, that is most of the budget.
    """
    sizes = np.bincount(labels, minlength=n_clusters).astype(np.float64)
    sizes[np.unique(labels[:n_labelled])] = 0
    taken = np.zeros(len(features) - n_labelled, dtype=bool)
    picks: List[int] = []
    n_random = 0

    for _ in range(budget):
        if sizes.max() == 0:
            pick = int(rng.choice(np.flatnonzero(~taken)))
            n_random += 1
        else:
            cluster = int(sizes.argmax())
            members = np.flatnonzero(labels == cluster)
            pick = int(members[typicality(features[members], k_nn).argmax()]) - n_labelled
            sizes[cluster] = 0
        taken[pick] = True
        picks.append(pick)

    logger.info(f' dal rule: {budget - n_random} typical picks, {n_random} uniform-random picks')
    return picks


def _next_cluster(order: np.ndarray, step: int, n_left: np.ndarray) -> Optional[int]:
    """First cluster at or after position `step` (cyclically) that still has unselected members."""
    for offset in range(len(order)):
        cluster = int(order[(step + offset) % len(order)])
        if n_left[cluster] > 0:
            return cluster
    return None


def _select_authors(
        features: np.ndarray, labels: np.ndarray, n_labelled: int, budget: int, n_clusters: int,
        k_nn: int, min_cluster_size: int
    ) -> List[int]:
    """The authors' rule: clusters are sorted once by (fewest labelled points, largest size) and
    visited cyclically, one most-typical unlabelled point per visit.

    The official code crashes when a visited cluster has no member left, which happens as soon as the
    budget exceeds the number of clusters. Such a cluster is skipped here (as in the copy shipped with
    the UHerding repository). Clusters too small to estimate a density, which the rule drops, are only
    visited once every kept cluster is exhausted, so that exactly `budget` points can always be chosen.
    """
    sizes = np.bincount(labels, minlength=n_clusters)
    labelled_counts = np.bincount(labels[:n_labelled], minlength=n_clusters)
    cluster_ids = np.arange(n_clusters)

    def visiting_order(ids: np.ndarray) -> np.ndarray:
        return ids[np.lexsort((ids, -sizes[ids], labelled_counts[ids]))]

    kept = visiting_order(cluster_ids[sizes > min_cluster_size])
    dropped = visiting_order(cluster_ids[(sizes > 0) & (sizes <= min_cluster_size)])

    remaining = labels.copy()
    remaining[:n_labelled] = -1  # labelled points never compete for a pick
    n_left = np.bincount(remaining[remaining >= 0], minlength=n_clusters)
    picks: List[int] = []
    step = 0

    while len(picks) < budget:
        cluster = _next_cluster(kept, step, n_left)
        if cluster is None:
            cluster = _next_cluster(dropped, step, n_left)
        if cluster is None:
            break
        members = np.flatnonzero(remaining == cluster)
        pick = int(members[typicality(features[members], k_nn).argmax()])
        remaining[pick] = -1
        n_left[cluster] -= 1
        picks.append(pick - n_labelled)
        step += 1

    return picks


def typiclust_select(
        labelled_features: np.ndarray,
        pool_features: np.ndarray,
        budget: int,
        rng: np.random.Generator,
        *,
        selection: str = 'authors',
        k_nn: int = 20,
        min_cluster_size: int = 5,
        max_clusters: int = 500,
        l2_normalize: bool = True,
        minibatch_threshold: int = 50,
        minibatch_size: int = 5000,
        cluster_labels: Optional[np.ndarray] = None,
    ) -> List[int]:
    """Choose `budget` pool points: typical ones, spread over K = min(|L| + B, max_clusters) clusters.

    Args:
        labelled_features: (L, D) features of the labelled set, which joins the clustering.
        pool_features: (N, D) features of the candidates.
        budget: number of points to select; clamped to N.
        rng: local generator, the only source of randomness (k-means seed, 'dal' fallback picks).
        selection: 'authors' or 'dal', see competitor_config.py.
        cluster_labels: optional clustering of the concatenated [labelled; pool] features, in
            [0, K), replacing k-means. Used by the golden tests to compare the selection rules with
            the reference code under an identical clustering.

    Returns:
        pool-local indices, exactly min(budget, N) of them, distinct.
    """
    if selection not in SELECTIONS:
        raise ValueError(f'selection must be one of {SELECTIONS}, got {selection!r}')

    pool_features = np.asarray(pool_features, dtype=np.float32)
    labelled_features = np.asarray(labelled_features, dtype=np.float32).reshape(-1, pool_features.shape[1])
    n_labelled, n_pool = len(labelled_features), len(pool_features)
    budget = min(budget, n_pool)
    if budget <= 0:
        return []

    features = np.concatenate([labelled_features, pool_features])
    if not np.isfinite(features).all():
        raise ValueError('the features contain NaN or Inf')
    if l2_normalize:
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)

    n_clusters = min(n_labelled + budget, max_clusters, len(features))
    kmeans_seed = int(rng.integers(0, 2 ** 31 - 1))  # drawn even without k-means, to keep the stream fixed
    if cluster_labels is not None:
        cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
        if cluster_labels.shape != (len(features),) or cluster_labels.min() < 0 or cluster_labels.max() >= n_clusters:
            raise ValueError(f'cluster_labels must hold one id in [0, {n_clusters}) per point, got shape {cluster_labels.shape}')

    # One thread, like the rest of the benchmark (main.py pins OMP_NUM_THREADS=1, but only after torch is
    # imported). Besides avoiding oversubscription next to DataLoader workers, this keeps the threaded BLAS
    # inside scikit-learn from aborting the process next to torch's OpenMP runtime (seen on macOS/MPS).
    with threadpool_limits(limits=1):
        if cluster_labels is None:
            labels = cluster_features(features, n_clusters, kmeans_seed, minibatch_threshold, minibatch_size)
        else:
            labels = cluster_labels
        if selection == 'dal':
            return _select_dal(features, labels, n_labelled, budget, n_clusters, k_nn, rng)
        return _select_authors(features, labels, n_labelled, budget, n_clusters, k_nn, min_cluster_size)



class TypiClust(FeatureSourceMixin, ActiveLearner):

    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any]) -> None:

        params = dict(competitor_params['typiclust'])
        params['max_clusters'] = params.pop('max_clusters_by_dataset').get(ct_p['dataset_name'], params['max_clusters'])
        self.params = params

        # the name keys results.csv rows and checkpoints, so every variant that changes the numbers needs its own
        name = self.__class__.__name__
        if params['selection'] != 'authors': name += f'_{params["selection"]}'
        if ct_p.get('feature_source', 'model') != 'model': name += '_ssl'

        super().__init__(ct_p, t_p, name)


    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
        """Select the observations to label with TypiClust.

        Args:
            sample_unlab_subset: The candidate pool, a Subset of the unlabelled-train dataset.
            n_top_k_obs: The budget.

        Returns:
            Tuple[List[int], List[int]]: The selected positions in the pool, and the corresponding
                dataset indices. Fewer than n_top_k_obs only if the pool is smaller than the budget.
        """
        logger.info(' => Getting the labelled and candidate features')
        labelled_features, pool_features = self.get_features(sample_unlab_subset)
        logger.info(' DONE\n')

        logger.info(' => TypiClust selection')
        rng = derive_rng(self.dataset.dataset_id, self.ct_p['trial'], self.iter, self.strategy_name)
        selected = typiclust_select(labelled_features, pool_features, n_top_k_obs, rng, **self.params)
        topk_idx_obs = check_selection(selected, len(pool_features), n_top_k_obs)
        logger.info(' DONE\n')

        pool_indices = list(sample_unlab_subset.indices)  # type: ignore
        return topk_idx_obs, [pool_indices[idx] for idx in topk_idx_obs]

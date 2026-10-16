# ProbCover: O. Yehuda, A. Dekel, G. Hacohen and D. Weinshall, "Active Learning Through a Covering Lens",
# NeurIPS 2022, https://arxiv.org/abs/2205.11320. Official code (linked in the abstract):
# https://github.com/avihu111/TypiClust
#
# Ported, with changes, from dal-toolbox (https://github.com/dhuseljic/dal-toolbox @ 13ee9ca):
# dal_toolbox/active_learning/strategies/prob_cover.py, whose own header reads "It is currently not
# working correctly". dal-toolbox licence:
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
# That file is itself "partially from" the authors' code (MIT licence):
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
# Changes: pure functions over arrays plus a thin ActiveLearner wrapper; a dense boolean graph over the
# candidates instead of a pandas edge list; the paper's purity rule for the radius, computed exactly and
# per query (dal-toolbox never calls its own estimate_delta, and that one is wrong, see delta_from_purity);
# no loss of the last `n mod batch_size` rows of the graph; distinct picks once everything is covered; a
# local seeded generator; features from a configurable source.

import math

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from torch.utils.data import Subset

from ActiveLearner import ActiveLearner
from competitor_config import competitor_params
from strategies.competitors.common import FeatureSourceMixin, check_selection, cluster_features, derive_rng

from typing import Any, Dict, List, Optional, Tuple

import logging
logger = logging.getLogger(__name__)


PAPER_PURITY = 0.95


def _nearest_other_cluster_distance(features: torch.Tensor, labels: np.ndarray, block_size: int) -> np.ndarray:
    """For every point, the distance to the closest point with another cluster label (inf if none)."""
    labels_t = torch.from_numpy(labels)
    nearest = torch.empty(len(features))
    for start in range(0, len(features), block_size):
        distances = torch.cdist(features[start:start + block_size], features)
        distances[labels_t[start:start + block_size, None] == labels_t[None, :]] = float('inf')
        nearest[start:start + block_size] = distances.min(dim=1).values
    return nearest.numpy()


def delta_from_purity(features: np.ndarray, labels: np.ndarray, alpha: float, block_size: int = 1000) -> float:
    """The largest radius at which at least a fraction `alpha` of the balls are pure.

    A ball is the set of points at distance < delta from its centre (the centre included); it is pure when
    every member has the centre's label. The ball around x is therefore pure exactly when
    delta <= d(x), the distance from x to the nearest point of another cluster, so the answer is an order
    statistic of those distances: the exact solution of the paper's criterion, with no search grid.
    (dal-toolbox's unused estimate_delta searches a grid of [1, 20] and decides purity by assuming that each
    point's self-loop opens its row of the edge list, which row-major ordering does not guarantee.)
    """
    if not 0 < alpha <= 1:
        raise ValueError(f'the purity threshold must be in (0, 1], got {alpha}')
    features = np.ascontiguousarray(features, dtype=np.float32)
    nearest_other = _nearest_other_cluster_distance(torch.from_numpy(features), np.asarray(labels), block_size)
    n_points = len(nearest_other)
    n_pure_needed = int(math.ceil(alpha * n_points - 1e-9))
    delta = float(np.sort(nearest_other)[n_points - n_pure_needed])
    if not math.isfinite(delta):   # a single cluster: every ball is pure, so any radius qualifies
        delta = 2.0 * float(np.linalg.norm(features, axis=1).max())
    return delta


def _select_greedy_cover(
        features: np.ndarray, n_labelled: int, budget: int, delta: float, rng: np.random.Generator, block_size: int
    ) -> List[int]:
    """Greedy max coverage: repeatedly take the candidate whose delta-ball holds the most uncovered candidates.

    Points within delta of a labelled point start out covered. Ties go to the lowest index. When nothing
    uncovered is left, the remaining picks are a seeded random draw from the unselected candidates (the
    reference code repeats index 0 there, which is a duplicate and may be a labelled sample).
    """
    tensor = torch.from_numpy(features)
    labelled, pool = tensor[:n_labelled], tensor[n_labelled:]
    n_pool = len(pool)

    adjacency = np.empty((n_pool, n_pool), dtype=bool)  # adjacency[c, y]: y lies in the ball around c
    for start in range(0, n_pool, block_size):
        adjacency[start:start + block_size] = (torch.cdist(pool[start:start + block_size], pool) < delta).numpy()
    np.fill_diagonal(adjacency, True)

    uncovered = np.ones(n_pool, dtype=bool)
    for start in range(0, n_labelled, block_size):
        uncovered &= ~(torch.cdist(labelled[start:start + block_size], pool) < delta).any(dim=0).numpy()
    initially_covered = n_pool - int(uncovered.sum())

    gains = np.concatenate([(adjacency[start:start + block_size] & uncovered).sum(axis=1)
                            for start in range(0, n_pool, block_size)])
    selectable = np.ones(n_pool, dtype=bool)
    picks: List[int] = []

    while len(picks) < budget:
        best = int(np.where(selectable, gains, -1).argmax())
        if gains[best] <= 0 or not selectable[best]:
            break
        newly_covered = np.flatnonzero(adjacency[best] & uncovered)
        uncovered[newly_covered] = False
        gains -= adjacency[:, newly_covered].sum(axis=1)
        selectable[best] = False
        picks.append(best)

    n_random = budget - len(picks)
    if n_random:
        picks.extend(int(pick) for pick in rng.permutation(np.flatnonzero(selectable))[:n_random])

    logger.info(
        f' delta {delta:.4f}: the labelled set covers {initially_covered / n_pool:.1%} of the candidates, '
        f'{len(picks) - n_random} greedy picks, {n_random} random picks once everything was covered'
    )
    return picks


def probcover_select(
        labelled_features: np.ndarray,
        pool_features: np.ndarray,
        budget: int,
        rng: np.random.Generator,
        *,
        n_classes: int,
        delta: Optional[float] = None,
        purity_threshold: float = PAPER_PURITY,
        l2_normalize: bool = True,
        minibatch_threshold: int = 50,
        minibatch_size: int = 5000,
        block_size: int = 1000,
        cluster_labels: Optional[np.ndarray] = None,
    ) -> List[int]:
    """Choose `budget` pool points that cover the most not-yet-covered pool points with delta-balls.

    Args:
        labelled_features: (L, D) features of the labelled set; the points it covers are not worth querying.
        pool_features: (N, D) features of the candidates, which are also the points to cover.
        budget: number of points to select; clamped to N.
        rng: local generator, the only source of randomness (pseudo-label k-means, the random tail).
        n_classes: number of k-means pseudo-label clusters used to estimate delta (the dataset's classes).
        delta: ball radius, or None for the paper's purity rule (see delta_from_purity).
        purity_threshold: alpha of the purity rule.
        cluster_labels: optional pseudo-labels of the concatenated [labelled; pool] features, replacing k-means.

    Returns:
        pool-local indices, exactly min(budget, N) of them, distinct.
    """
    pool_features = np.asarray(pool_features, dtype=np.float32)
    labelled_features = np.asarray(labelled_features, dtype=np.float32).reshape(-1, pool_features.shape[1])
    n_labelled, n_pool = len(labelled_features), len(pool_features)
    budget = min(budget, n_pool)
    if budget <= 0:
        return []
    if delta is not None and not (math.isfinite(delta) and delta > 0):
        raise ValueError(f'delta must be a positive number, got {delta}')

    features = np.concatenate([labelled_features, pool_features])
    if not np.isfinite(features).all():
        raise ValueError('the features contain NaN or Inf')
    if l2_normalize:
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)

    kmeans_seed = int(rng.integers(0, 2 ** 31 - 1))  # drawn even when unused, to keep the stream fixed
    if delta is None:
        if cluster_labels is None:
            with threadpool_limits(limits=1):
                cluster_labels = cluster_features(features, min(n_classes, len(features)), kmeans_seed, minibatch_threshold, minibatch_size)
        delta = delta_from_purity(features, cluster_labels, purity_threshold, block_size)

    return _select_greedy_cover(features, n_labelled, budget, delta, rng, block_size)



class ProbCover(FeatureSourceMixin, ActiveLearner):

    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any]) -> None:

        self.params = dict(competitor_params['probcover'])

        # the name keys results.csv rows and checkpoints, so every variant that changes the numbers needs its own
        name = self.__class__.__name__
        if self.params['delta'] is not None: name += f'_d{self.params["delta"]:g}'
        elif self.params['purity_threshold'] != PAPER_PURITY: name += f'_a{self.params["purity_threshold"]:g}'
        if ct_p.get('feature_source', 'model') != 'model': name += '_ssl'

        super().__init__(ct_p, t_p, name)


    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
        """Select the observations to label with ProbCover.

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

        logger.info(' => ProbCover selection')
        rng = derive_rng(self.dataset.dataset_id, self.ct_p['trial'], self.iter, self.strategy_name)
        selected = probcover_select(
            labelled_features, pool_features, n_top_k_obs, rng, n_classes=self.dataset.n_classes, **self.params
        )
        topk_idx_obs = check_selection(selected, len(pool_features), n_top_k_obs)
        logger.info(' DONE\n')

        pool_indices = list(sample_unlab_subset.indices)  # type: ignore
        return topk_idx_obs, [pool_indices[idx] for idx in topk_idx_obs]

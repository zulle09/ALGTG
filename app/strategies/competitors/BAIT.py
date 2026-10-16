# BAIT: J. T. Ash, S. Goel, A. Krishnamurthy and S. Kakade, "Gone Fishing: Neural Active Learning with
# Fisher Embeddings", NeurIPS 2021, https://arxiv.org/abs/2106.09675. The authors' classification code is in
# https://github.com/JordanAsh/badge, which ships no licence file.
#
# Implemented from the paper's equations (App. A.2, the Fisher information of a softmax last layer; Alg. 1,
# the forward-backward greedy on the trace objective with the Woodbury identity), not translated from
# dal-toolbox's bait.py. That file (BSD-3-Clause, Copyright (c) 2020 The dal_toolbox developers) carries no
# attribution but mirrors the licence-less JordanAsh/badge code identifier by identifier and comment by
# comment, so a translation would inherit that provenance. dal-toolbox @ 13ee9ca is used only as a numerical
# oracle in tests/golden/make_bait_golden.py; nothing from it is copied.
#
# The formulation differs from both: a sample's Fisher factor is the Kronecker product X_i = A_i (x) phi_i^T,
# so the (C x C*D) embeddings are never built, and the per-candidate score matrices are updated by rank-r
# corrections instead of being recomputed against the (C*D x C*D) inverse at every step.

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ActiveLearner import ActiveLearner
from competitor_config import competitor_params
from strategies.competitors.common import check_selection

from typing import Any, Dict, List, Tuple

import logging
logger = logging.getLogger(__name__)


FISHERS = ('ce', 'bce')
PAPER_LAMBDA = 1.0
PAPER_LAMBDA_BY_DATASET = {'cifar10': 0.01}
PAPER_OVERSAMPLE = 2


def fisher_factors(logits: np.ndarray, kind: str) -> np.ndarray:
    """A_i for every sample, such that X_i = A_i (x) phi_i^T and the sample's Fisher information is X_i^T X_i.

    ce:  the softmax last layer. A_i[j, h] = sqrt(p_j) (delta_jh - p_h), shape (C, C), and
         A_i^T A_i = diag(p) - p p^T, so X_i^T X_i = (diag(p) - p p^T) (x) phi phi^T (App. A.2).
    bce: dal-toolbox's cheap alternative, a Bernoulli likelihood on the top probability m = max p. A_i is
         (2, 1): [sqrt(m) (1 - m), -sqrt(1 - m) m], so X_i^T X_i = m (1 - m) phi phi^T and the dimension is D
         instead of C*D. It is an approximation, not the paper's Fisher.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    if kind == 'ce':
        return np.sqrt(probs)[:, :, None] * (np.eye(probs.shape[1])[None] - probs[:, None, :])
    top = probs.max(axis=1)
    return np.stack([np.sqrt(top) * (1 - top), -np.sqrt(1 - top) * top], axis=1)[:, :, None]


def fisher_sum(factors: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """sum_i X_i^T X_i as a (c*D, c*D) matrix, with index h*D + k for class block h and feature k.

    X_i^T X_i = (A_i^T A_i) (x) (phi_i phi_i^T), so the (h, g) block is sum_i (A_i^T A_i)[h, g] phi_i phi_i^T.
    """
    n_classes, dim = factors.shape[2], features.shape[1]
    gram = torch.einsum('njh,njg->nhg', factors, factors)
    result = torch.zeros(n_classes, dim, n_classes, dim, dtype=features.dtype, device=features.device)
    for h in range(n_classes):
        for g in range(h, n_classes):
            block = features.T @ (gram[:, h, g, None] * features)
            result[h, :, g, :] = block
            result[g, :, h, :] = block   # the block is symmetric, and the (g, h) one equals it
    return result.reshape(n_classes * dim, n_classes * dim)


def quadratic_forms(matrix: torch.Tensor, factors: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """X_i W X_i^T, an (r, r) matrix per sample, for a symmetric (c*D, c*D) matrix W."""
    n_classes, dim = factors.shape[2], features.shape[1]
    blocks = matrix.reshape(n_classes, dim, n_classes, dim)
    inner = torch.empty(len(features), n_classes, n_classes, dtype=features.dtype, device=features.device)
    for h in range(n_classes):
        for g in range(h, n_classes):
            inner[:, h, g] = inner[:, g, h] = ((features @ blocks[h, :, g, :]) * features).sum(dim=1)
    return factors @ inner @ factors.transpose(1, 2)


def project(factors: torch.Tensor, features: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
    """X_i V for a (c*D, m) matrix V, an (r, m) matrix per sample."""
    n_classes, dim = factors.shape[2], features.shape[1]
    blocks = columns.reshape(n_classes, dim, -1)
    inner = (features @ blocks.permute(1, 0, 2).reshape(dim, -1)).reshape(len(features), n_classes, -1)
    return factors @ inner


def check_feasible(fisher: str, n_classes: int, embedding_dim: int, max_embedding_dim: int) -> int:
    """The dimension of the Fisher matrix, or a refusal when it would not fit in memory."""
    if fisher not in FISHERS:
        raise ValueError(f'fisher must be one of {FISHERS}, got {fisher!r}')
    dimension = n_classes * embedding_dim if fisher == 'ce' else embedding_dim
    if dimension > max_embedding_dim:
        raise ValueError(
            f'the exact BAIT Fisher matrix is {dimension} x {dimension} ({n_classes} classes x {embedding_dim} features), '
            f'above max_embedding_dim={max_embedding_dim}; use fisher="bce" (dimension {embedding_dim}) for this dataset'
        )
    return dimension


def bait_select(
        labelled_logits: np.ndarray,
        labelled_features: np.ndarray,
        pool_logits: np.ndarray,
        pool_features: np.ndarray,
        budget: int,
        *,
        fisher: str = 'ce',
        lambda_reg: float = PAPER_LAMBDA,
        oversample: int = PAPER_OVERSAMPLE,
        max_embedding_dim: int = 12000,
        device: str = 'cpu',
    ) -> List[int]:
    """Choose `budget` pool points that minimise tr((sum of their Fisher information + prior)^-1 F).

    F is the mean Fisher information of all samples, labelled and candidates; the prior is lambda I plus the
    labelled samples' mean Fisher weighted by L / (L + B); candidates enter with weight B / (L + B), as in the
    reference implementation. A forward greedy pass adds `oversample` * B candidates, each the one that lowers
    the trace most, then a backward pass removes the ones whose removal costs least until B remain.

    Args:
        labelled_logits, labelled_features: (L, C) logits and (L, D) penultimate features of the labelled set.
        pool_logits, pool_features: the same for the N candidates.
        budget: number of points to select; clamped to N.
        fisher: 'ce' (exact softmax Fisher, dimension C*D) or 'bce' (dal-toolbox's approximation, dimension D).
        device: 'cpu' or 'cuda'. The linear algebra runs in float64.

    Returns:
        pool-local indices, exactly min(budget, N) of them, distinct. Ties go to the lowest index.
    """
    pool_features = np.asarray(pool_features, dtype=np.float64)
    pool_logits = np.asarray(pool_logits, dtype=np.float64)
    n_pool, dim = pool_features.shape
    labelled_features = np.asarray(labelled_features, dtype=np.float64).reshape(-1, dim)
    n_labelled = len(labelled_features)
    labelled_logits = np.asarray(labelled_logits, dtype=np.float64)
    if n_labelled: labelled_logits = labelled_logits.reshape(n_labelled, -1)
    if fisher not in FISHERS:
        raise ValueError(f'fisher must be one of {FISHERS}, got {fisher!r}')
    if not lambda_reg > 0:
        raise ValueError(f'lambda_reg must be positive, got {lambda_reg}')
    if oversample < 1:
        raise ValueError(f'oversample must be at least 1, got {oversample}')
    for name, array in (('features', pool_features), ('logits', pool_logits), ('labelled features', labelled_features), ('labelled logits', labelled_logits)):
        if not np.isfinite(array).all():
            raise ValueError(f'the {name} contain NaN or Inf')

    budget = min(budget, n_pool)
    if budget <= 0:
        return []
    if budget == n_pool:
        return list(range(n_pool))

    pool_factors = fisher_factors(pool_logits, fisher)
    rank = pool_factors.shape[1]
    dimension = check_feasible(fisher, pool_logits.shape[1], dim, max_embedding_dim)

    as_tensor = lambda array: torch.as_tensor(array, dtype=torch.float64, device=device)
    pool_factors, pool_features = as_tensor(pool_factors), as_tensor(pool_features)
    labelled_factors = as_tensor(fisher_factors(labelled_logits, fisher)) if n_labelled else pool_factors[:0]
    labelled_features = as_tensor(labelled_features) if n_labelled else pool_features[:0]

    labelled_fisher = fisher_sum(labelled_factors, labelled_features) if n_labelled else torch.zeros(dimension, dimension, dtype=torch.float64, device=device)
    mean_fisher = (labelled_fisher + fisher_sum(pool_factors, pool_features)) / (n_labelled + n_pool)
    prior = lambda_reg * torch.eye(dimension, dtype=torch.float64, device=device) + labelled_fisher / (n_labelled + budget)
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(prior))                       # M^-1, kept up to date by rank-r updates
    candidate_factors = pool_factors * np.sqrt(budget / (n_labelled + budget))

    own_quad = quadratic_forms(inverse, candidate_factors, pool_features)                 # X_i M^-1 X_i^T
    fisher_quad = quadratic_forms(inverse @ mean_fisher @ inverse, candidate_factors, pool_features)   # X_i M^-1 F M^-1 X_i^T
    identity = torch.eye(rank, dtype=torch.float64, device=device)

    def scores(rows, sign):
        """tr((sign I + own)^-1 fisher) per candidate: the decrease of the trace objective if it were added (sign +1), or
        minus the increase if it were removed (sign -1, so the removal that costs least has the largest score)."""
        own, other = (own_quad, fisher_quad) if rows is None else (own_quad[rows], fisher_quad[rows])
        scored = torch.linalg.solve(sign * identity + own, other).diagonal(dim1=-2, dim2=-1).sum(-1).cpu().numpy()
        if not np.isfinite(scored).all():
            raise FloatingPointError('BAIT scores are not finite: the Fisher matrix is ill-conditioned, increase lambda_reg')
        return scored

    def commit(index, sign, rows):
        """Add (sign +1) or remove (sign -1) a candidate: update M^-1 and the tracked matrices of `rows` (None: all)."""
        nonlocal inverse
        embedding_t = torch.einsum('jh,k->hkj', candidate_factors[index], pool_features[index]).reshape(dimension, rank)
        t = inverse @ embedding_t
        middle = torch.linalg.inv(sign * identity + own_quad[index])
        r = mean_fisher @ t
        z = inverse @ r
        t_r = t.T @ r
        factors, features = (candidate_factors, pool_features) if rows is None else (candidate_factors[rows], pool_features[rows])
        x_t, x_z = project(factors, features, t), project(factors, features, z)
        left = x_t @ middle
        cross = left @ x_z.transpose(1, 2)
        own_delta = left @ x_t.transpose(1, 2)
        fisher_delta = left @ t_r @ left.transpose(1, 2) - cross - cross.transpose(1, 2)
        if rows is None:
            own_quad.sub_(own_delta)
            fisher_quad.add_(fisher_delta)
        else:
            own_quad[rows] -= own_delta
            fisher_quad[rows] += fisher_delta
        inverse.addmm_(t @ middle, t.T, alpha=-1.0)

    selected: List[int] = []
    available = np.ones(n_pool, dtype=bool)
    for _ in range(min(oversample * budget, n_pool)):
        gains = np.where(available, scores(None, +1), -np.inf)
        best = int(gains.argmax())                      # first maximum: the lowest index wins a tie
        selected.append(best)
        available[best] = False
        commit(best, +1, None)

    while len(selected) > budget:
        rows = torch.as_tensor(selected, dtype=torch.long, device=device)
        position = int(scores(rows, -1).argmax())       # the removal that costs least; first in selection order on a tie
        removed = selected.pop(position)
        commit(removed, -1, torch.as_tensor(selected, dtype=torch.long, device=device))

    return selected



class BAIT(ActiveLearner):

    fisher = 'ce'

    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any]) -> None:

        if ct_p.get('feature_source', 'model') != 'model':
            raise ValueError("BAIT is the Fisher information of the trained model's own last layer: it needs that model's "
                             'penultimate features and logits, so -fs file is not supported')

        params = dict(competitor_params['bait'])
        params['fisher'] = self.fisher
        params['lambda_reg'] = params.pop('lambda_by_dataset').get(ct_p['dataset_name'], params['lambda_reg'])
        check_feasible(params['fisher'], ct_p['Dataset'].n_classes, ct_p['Master_Model'].backbone.get_embedding_dim(), params['max_embedding_dim'])
        self.params = params

        # the name keys results.csv rows and checkpoints, so every variant that changes the numbers needs its own
        name = 'BAIT'
        if self.fisher != 'ce': name += f'_{self.fisher}'
        if params['lambda_reg'] != PAPER_LAMBDA_BY_DATASET.get(ct_p['dataset_name'], PAPER_LAMBDA): name += f'_l{params["lambda_reg"]:g}'
        if params['oversample'] != PAPER_OVERSAMPLE: name += f'_o{params["oversample"]}'

        super().__init__(ct_p, t_p, name)


    def _outputs(self, subset: Subset) -> Tuple[np.ndarray, np.ndarray]:
        """Penultimate features and logits of the current model for a subset, as float32 arrays."""
        dataloader = DataLoader(subset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        outputs = {
            'embedds': torch.empty((0, self.model.backbone.get_embedding_dim()), dtype=torch.float32),
            'outs': torch.empty((0, self.dataset.n_classes), dtype=torch.float32),
        }
        self.get_embeddings(dataloader, outputs, embedds2cpu=True)
        return outputs['embedds'].numpy(), outputs['outs'].numpy()


    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
        """Select the observations to label with BAIT.

        Args:
            sample_unlab_subset: The candidate pool, a Subset of the unlabelled-train dataset.
            n_top_k_obs: The budget.

        Returns:
            Tuple[List[int], List[int]]: The selected positions in the pool, and the corresponding
                dataset indices. Fewer than n_top_k_obs only if the pool is smaller than the budget.
        """
        logger.info(' => Getting the labelled and candidate features and logits')
        self.load_best_checkpoint()
        labelled_features, labelled_logits = self._outputs(Subset(self.dataset.unlab_train_ds, self.labelled_indices))
        pool_features, pool_logits = self._outputs(sample_unlab_subset)
        logger.info(' DONE\n')

        logger.info(' => BAIT selection')
        selected = bait_select(labelled_logits, labelled_features, pool_logits, pool_features, n_top_k_obs, **self.params)
        topk_idx_obs = check_selection(selected, len(pool_features), n_top_k_obs)
        logger.info(' DONE\n')

        pool_indices = list(sample_unlab_subset.indices)  # type: ignore
        return topk_idx_obs, [pool_indices[idx] for idx in topk_idx_obs]



class BAIT_bce(BAIT):
    """BAIT with dal-toolbox's Bernoulli Fisher: rank 2 and dimension D for any number of classes. An approximation."""

    fisher = 'bce'

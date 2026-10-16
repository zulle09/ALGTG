
import zlib

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from typing import Dict, List, Sequence, Tuple

import logging
logger = logging.getLogger(__name__)


FEATURE_SOURCES = ('model', 'file')


def derive_rng(dataset_id: int, trial: int, cycle: int, salt: str) -> np.random.Generator:
    """Local generator for one query.

    A competitor never reads or advances the global RNGs: whatever the preceding training consumed,
    the same (dataset, trial, cycle, strategy name) always yields the same stream.
    """
    return np.random.default_rng([dataset_id, trial, cycle, zlib.crc32(salt.encode('utf-8'))])


def check_selection(selected: Sequence[int], n_candidates: int, budget: int) -> List[int]:
    """Refuse a selection the AL loop cannot use instead of repairing it.

    `ActiveLearner.update_sets` removes the returned indices from the candidate list one by one, so a
    duplicate or foreign index kills the run late. A budget larger than the pool is the only case in
    which fewer than `budget` indices are valid.
    """
    picked = [int(index) for index in selected]
    expected = min(budget, n_candidates)
    if len(picked) != expected:
        raise ValueError(f'expected {expected} indices, got {len(picked)}')
    if len(set(picked)) != len(picked):
        raise ValueError('the selection contains duplicate indices')
    if picked and (min(picked) < 0 or max(picked) >= n_candidates):
        raise ValueError(f'the selection has an index outside the {n_candidates} candidates')
    return picked


def load_feature_table(path: str, n_samples: int) -> np.ndarray:
    """Load externally computed features: one row per training sample, in `train_ds` order."""
    table = np.load(path)
    if table.ndim != 2 or table.shape[0] != n_samples:
        raise ValueError(f'{path}: expected one feature row per training sample, i.e. ({n_samples}, D), got {table.shape}')
    if not np.isfinite(table).all():
        raise ValueError(f'{path}: the features contain NaN or Inf')
    return table.astype(np.float32, copy=False)


class FeatureSourceMixin:
    """Where a competitor reads its features from (`-fs/--feature_source`).

    model: the penultimate embedding of the model trained in the current cycle, the same features
           CoreSet and BADGE use. This is the default.
    file:  a `.npy` of precomputed features (`-fp`), e.g. self-supervised features, which is what
           TypiClust, ProbCover and the herding papers use.
    Mixed in before ActiveLearner, which provides everything accessed here.
    """

    feature_table: np.ndarray | None = None

    def _model_features(self, subset: Subset) -> np.ndarray:
        dataloader = DataLoader(subset, batch_size=self.batch_size, shuffle=False, pin_memory=True)  # type: ignore
        embeddings: Dict[str, torch.Tensor] = {
            'embedds': torch.empty((0, self.model.backbone.get_embedding_dim()), dtype=torch.float32)  # type: ignore
        }
        self.get_embeddings(dataloader, embeddings, embedds2cpu=True)  # type: ignore
        return embeddings['embedds'].numpy()


    def get_features(self, pool_subset: Subset) -> Tuple[np.ndarray, np.ndarray]:
        """Features of the labelled set and of the candidate pool, as float32 arrays (n, D)."""
        pool_indices = list(pool_subset.indices)  # type: ignore

        if self.ct_p.get('feature_source', 'model') == 'file':  # type: ignore
            if self.feature_table is None:
                self.feature_table = load_feature_table(self.ct_p['feature_path'], len(self.dataset.train_ds))  # type: ignore
            return self.feature_table[self.labelled_indices], self.feature_table[pool_indices]  # type: ignore

        # the query must score with this strategy's own model, not whichever trained last
        self.load_best_checkpoint()  # type: ignore
        labelled_features = self._model_features(Subset(self.dataset.unlab_train_ds, self.labelled_indices))  # type: ignore
        return labelled_features, self._model_features(pool_subset)

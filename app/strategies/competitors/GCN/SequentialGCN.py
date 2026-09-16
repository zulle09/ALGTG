
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ActiveLearner import ActiveLearner
from strategies.competitors.GCN.gcn_model import GCN, aff_to_adj, bce_adj_loss

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)


class SequentialGCN(ActiveLearner):
    """Sequential GCN for Active Learning (Caramalau et al., CVPR 2021).

    A GCN is fitted at query time over a graph whose nodes are the labelled and the sampled
    unlabelled observations, and whose edges are cosine affinities between backbone features.
    It is trained to separate labelled from unlabelled nodes; the acquisition then reads off
    that model in one of two ways, selected by the `SELECTION` class attribute:

        'uncertain' (UncertainGCN) -- query the unlabelled nodes whose score is closest to
                                      `s_margin`, i.e. the ones the GCN cannot place.
        'core'      (CoreGCN)      -- run k-center greedy in the GCN hidden space, seeded
                                      with the labelled nodes.

    This is the closest graph-based competitor to ALGTG and is included for that reason: it
    also builds a graph over labelled+unlabelled samples, but *learns* node embeddings, whereas
    ALGTG derives its signal from the dynamics of a transduction game on a fixed payoff.
    """

    SELECTION: str = 'uncertain'

    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any]) -> None:
        super().__init__(ct_p, t_p, self.__class__.__name__)

        # Hyper-parameters from the reference implementation.
        self.gcn_hidden = 128
        self.gcn_dropout = 0.3
        self.gcn_epochs = 200
        self.gcn_lr = 1e-3
        self.gcn_wd = 5e-4
        self.gcn_lambda = 1.2      # weight of the unlabelled term in the BCE loss
        self.s_margin = 0.1        # target score for UncertainGCN


    def _k_center_greedy(self, feat_unlab: torch.Tensor, feat_lab: torch.Tensor, k: int) -> List[int]:
        """k-center greedy seeded with the labelled nodes (CoreGCN selection)."""
        if feat_lab.size(0) == 0:
            min_dist = torch.full((feat_unlab.size(0),), float('inf'), device=feat_unlab.device)
        else:
            min_dist = torch.cdist(feat_unlab, feat_lab).min(dim=1).values

        picked: List[int] = []
        while len(picked) < k:
            idx = int(torch.argmax(min_dist).item())
            picked.append(idx)
            min_dist = torch.minimum(min_dist, torch.cdist(feat_unlab, feat_unlab[idx].unsqueeze(0))[:, 0])
            # Explicitly retire the chosen point: when the remaining points all sit at distance 0
            # (collapsed features) argmax would hand back an already-picked index.
            min_dist[idx] = float('-inf')
        return picked


    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
        """Fit the GCN over the labelled+unlabelled graph and select the top-k unlabelled nodes.

        Args:
            sample_unlab_subset: the sampled unlabelled pool for this AL iteration.
            n_top_k_obs: how many observations to query.

        Returns:
            (positions within `self.rand_unlab_sample`, the corresponding dataset indices).
        """
        dl_dict = dict(batch_size=self.batch_size, shuffle=False, pin_memory=True)
        unlab_train_dl = DataLoader(sample_unlab_subset, **dl_dict)
        lab_train_dl = DataLoader(Subset(self.dataset.unlab_train_ds, self.labelled_indices), **dl_dict)

        logger.info(' => Getting the labelled and unlabelled embeddings')
        emb_dim = self.model.backbone.get_embedding_dim()
        unlab_embedds_dict = {'embedds': torch.empty((0, emb_dim), dtype=torch.float32, device=self.device)}
        lab_embedds_dict = {'embedds': torch.empty((0, emb_dim), dtype=torch.float32, device=self.device)}

        self.load_best_checkpoint()
        self.get_embeddings(unlab_train_dl, unlab_embedds_dict)
        self.get_embeddings(lab_train_dl, lab_embedds_dict)
        logger.info(' DONE\n')

        # Unlabelled nodes come first, so score/feature row i maps to rand_unlab_sample[i].
        n_unlab = unlab_embedds_dict["embedds"].size(0)
        # .clone() is required: get_embeddings runs under torch.inference_mode(), and inference
        # tensors cannot be saved for backward — feeding them straight into the GCN would raise
        # at loss.backward().
        features = torch.cat(
            (unlab_embedds_dict["embedds"], lab_embedds_dict["embedds"]), dim=0
        ).detach().clone()
        features = F.normalize(features, p=2, dim=1)

        n_nodes = features.size(0)
        logger.info(f' => Building the {n_nodes} x {n_nodes} propagation matrix '
                    f'(~{n_nodes * n_nodes * 4 / 1e9:.2f} GB in float32)')
        adj = aff_to_adj(features)

        lbl_mask = torch.zeros(n_nodes, dtype=torch.bool, device=self.device)
        lbl_mask[n_unlab:] = True

        gcn = GCN(nfeat=features.size(1), nhid=self.gcn_hidden, nclass=1, dropout=self.gcn_dropout).to(self.device)
        optimizer = torch.optim.Adam(gcn.parameters(), lr=self.gcn_lr, weight_decay=self.gcn_wd)

        logger.info(f' => Training the GCN for {self.gcn_epochs} epochs')
        gcn.train()
        for epoch in range(self.gcn_epochs):
            optimizer.zero_grad(set_to_none=True)
            scores, _ = gcn(features, adj)
            loss = bce_adj_loss(scores.squeeze(1), lbl_mask, ~lbl_mask, self.gcn_lambda)
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0: logger.info(f'  GCN epoch {epoch} | loss {loss.item():.4f}')
        logger.info(' DONE\n')

        gcn.eval()
        with torch.no_grad():
            scores, feat = gcn(features, adj)
        scores = scores.squeeze(1)

        logger.info(f' => Extracting the Top-k unlabelled observations ({self.SELECTION})')
        if self.SELECTION == 'uncertain':
            # closest to s_margin == the nodes the GCN is least able to place
            overall_topk = torch.topk(
                torch.abs(scores[:n_unlab] - self.s_margin), n_top_k_obs, largest=False
            ).indices.tolist()
        else:
            overall_topk = self._k_center_greedy(feat[:n_unlab], feat[n_unlab:], n_top_k_obs)
        logger.info(' DONE\n')

        del adj, features
        torch.cuda.empty_cache()

        return overall_topk, [self.rand_unlab_sample[id] for id in overall_topk]



class UncertainGCN(SequentialGCN):
    """Sequential GCN with the uncertainty-based selection."""
    SELECTION = 'uncertain'


class CoreGCN(SequentialGCN):
    """Sequential GCN with the k-center-greedy selection in GCN feature space."""
    SELECTION = 'core'

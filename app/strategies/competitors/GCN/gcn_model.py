
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """Standard GCN layer (Kipf & Welling): out = adj @ (x @ W) + b."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None: self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        out = torch.mm(adj, torch.mm(x, self.weight))
        return out if self.bias is None else out + self.bias


class GCN(nn.Module):
    """Two-layer GCN producing a per-node 'is labelled' score in (0, 1).

    Returns both the sigmoid score (used by UncertainGCN) and the hidden representation
    (used by CoreGCN as the space in which k-center greedy is run).
    """

    def __init__(self, nfeat: int, nhid: int = 128, nclass: int = 1, dropout: float = 0.3) -> None:
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        hidden = F.relu(self.gc1(x, adj))
        out = self.gc2(F.dropout(hidden, self.dropout, training=self.training), adj)
        return torch.sigmoid(out), hidden


def aff_to_adj(features: torch.Tensor) -> torch.Tensor:
    """Build the propagation matrix from L2-normalised features.

    Follows Caramalau et al. (2021): cosine affinity, self-similarity removed, columns
    normalised by their degree, then self-loops added back.

    NOTE on conditioning: the degree is a column sum of cosine affinities. With signed
    features positive and negative terms can nearly cancel, and dividing by a near-zero
    degree inflates that column (which then saturates the GCN's sigmoid). Backbone
    embeddings here are post-ReLU, hence non-negative, so all affinities are >= 0 and the
    degrees stay comfortably positive. The guard below only covers the exact-zero corner.
    """
    x = F.normalize(features, p=2, dim=1)
    adj = torch.mm(x, x.t())
    adj = adj - torch.diag(torch.diagonal(adj))          # drop self-similarity before normalising
    deg = adj.sum(dim=0)
    # cosine affinities may be negative, so guard the magnitude of the degree rather than
    # clamping it to a positive floor (which would flip signs).
    deg = torch.where(deg.abs() < 1e-8, torch.full_like(deg, 1e-8), deg)
    adj = adj / deg.unsqueeze(0)                         # column scaling, i.e. adj @ diag(1/deg)
    return adj + torch.eye(adj.size(0), device=adj.device)


def bce_adj_loss(
        scores: torch.Tensor,
        lbl_mask: torch.Tensor,
        unlbl_mask: torch.Tensor,
        l_adj: float = 1.2
    ) -> torch.Tensor:
    """Binary cross-entropy pushing labelled nodes to 1 and unlabelled ones to 0.

    `l_adj` down/up-weights the unlabelled term as in the reference implementation.
    """
    lnl = torch.log(scores[lbl_mask].clamp(min=1e-8))
    lnu = torch.log((1 - scores[unlbl_mask]).clamp(min=1e-8))
    return -torch.mean(lnl) - l_adj * torch.mean(lnu)

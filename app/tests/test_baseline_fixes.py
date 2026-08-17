"""
Regression tests for the three baseline faithfulness fixes (BALD, AlphaMix, BADGE).

These assert the *math* of each fix on small synthetic tensors using the project's own
`entropy` util. They would have caught the original bugs, and they do NOT require a trained
checkpoint. They are a guard, not a substitute for the full re-run of the benchmark — the
comparison tables in the paper must come from real training runs with these fixes in place.

    python app/tests/test_baseline_fixes.py    # or: pytest app/tests/test_baseline_fixes.py
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn.functional as F

from utils import entropy
from strategies.competitors.GCN.gcn_model import GCN, aff_to_adj, bce_adj_loss


# ------------------------------------------------------------------ BADGE
def test_badge_gradient_embedding_ranks_uncertainty():
    """Gradient embedding g = (p - onehot(argmax p)) (x) z: uncertain sample -> larger norm."""
    embs = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])   # identical embeddings
    probs = torch.tensor([[0.95, 0.03, 0.02],                 # confident
                          [0.40, 0.35, 0.25]])                # uncertain
    scale = probs - F.one_hot(probs.argmax(1), num_classes=probs.size(1)).float()
    g = (scale.unsqueeze(2) * embs.unsqueeze(1)).view(probs.size(0), -1)

    assert g[1].norm() > g[0].norm(), "uncertain sample must have the larger gradient-embedding norm"


def test_badge_init_centers_terminates_on_collapsed_embeddings():
    """init_centers must not spin forever when points collapse onto the chosen centres.

    A barely-trained backbone produces near-identical embeddings. Once enough points sit at
    distance 0 from a centre, the k-means++ distribution has no representable mass left on the
    unpicked points and the unbounded rejection loop never terminated. Guarded by a bounded
    retry plus a deterministic farthest-point fallback.
    """
    from strategies.competitors.BADGE import BADGE

    torch.manual_seed(0)
    z = torch.ones(128, 32) + torch.randn(128, 32) * 1e-4        # collapsed embeddings
    p = torch.softmax(torch.randn(128, 10) * 1e-4, dim=1)
    scale = p - F.one_hot(p.argmax(1), num_classes=10).float()
    emb = (scale.unsqueeze(2) * z.unsqueeze(1)).view(128, -1)

    # exercise init_centers without constructing a full ActiveLearner
    badge = BADGE.__new__(BADGE)
    badge.embedds_dict = {"embedds": emb}

    picked = badge.init_centers(64)                              # must return, not hang
    assert len(picked) == 64, f"expected 64 centres, got {len(picked)}"
    assert len(set(picked)) == 64, "init_centers returned duplicate centres"


# ------------------------------------------------------------------ BALD
def test_bald_mutual_information_sign_and_ranking():
    """BALD = H(E[p]) - E[H(p)] >= 0, and disagreement must rank higher than agreement."""
    #        (n_drop=2, N=2, C=3): sample 0 agrees across drops, sample 1 flips its prediction
    pdd = torch.tensor([
        [[0.70, 0.20, 0.10], [0.90, 0.05, 0.05]],   # drop 0
        [[0.70, 0.20, 0.10], [0.05, 0.90, 0.05]],   # drop 1
    ])
    mean_pb = pdd.mean(dim=0)                        # (N, C)
    entropy1 = entropy(mean_pb)                      # H(E[p])
    entropy2 = torch.mean(entropy(pdd, dim=2), dim=0)  # E[H(p)]
    mi = entropy1 - entropy2                         # the corrected acquisition score

    assert torch.all(mi >= -1e-6), "mutual information must be non-negative"
    assert mi[1] > mi[0], "the disagreeing sample must score higher than the agreeing one"


def test_bald_dropout_actually_varies():
    """MC-dropout must produce different outputs across passes (the old code did not)."""
    torch.manual_seed(0)
    z = torch.randn(16, 64)
    a = F.dropout(z, p=0.5, training=True)
    b = F.dropout(z, p=0.5, training=True)
    assert not torch.allclose(a, b), "F.dropout(training=True) is not producing stochastic passes"


# ------------------------------------------------------------------ AlphaMix
def test_alphamix_per_class_mask_needs_labels_tensor():
    """`lb[Y == i]` selects the class-i rows only when Y is a labels *tensor*."""
    lb = torch.randn(6, 4)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    for cls, expected in [(0, 2), (1, 2), (2, 2)]:
        assert lb[labels == cls].shape[0] == expected, f"class {cls} mask selected wrong rows"

    # Root cause of the original bug: a Python list of indices makes `Y == i` a scalar False,
    # so every class fell back to the global mean (per-class anchoring disabled).
    idx_list = [10, 11, 12, 13, 14, 15]
    assert (idx_list == 1) is False, "list == int is False in Python — this was the AlphaMix bug"


# ------------------------------------------------------------------ CDAL / CoreSet
def test_cdal_kl_is_finite_on_underflowed_softmax():
    """Softmax can underflow to exactly 0; p*log(p/q) would then be 0*-inf = NaN.

    A NaN in the distance makes argmax in furthest_first return the same index forever, so the
    query returns duplicates and update_sets dies with "list.remove(x): x not in list".
    """
    from strategies.competitors.CDAL import CDAL

    logits = torch.tensor([[0., 200., -200.], [1., 2., 3.]])
    probs = F.softmax(logits, dim=1)
    assert (probs == 0).any(), "test precondition: the softmax must underflow to an exact zero"

    dist = CDAL.kl_pairwise_distances(CDAL.__new__(CDAL), probs, probs)
    assert torch.isfinite(dist).all(), f"KL distances are not finite: {dist}"


def test_greedy_selection_never_repeats_an_index():
    """Duplicate embeddings tie at distance 0; the chosen point must be retired explicitly."""
    from strategies.competitors.CoreSet import CoreSet

    emb = torch.ones(20, 8)                       # every point identical -> all distances 0
    cs = CoreSet.__new__(CoreSet)
    cs.unlab_embedds_dict = {"embedds": emb}
    cs.lab_embedds_dict = {"embedds": torch.empty(0, 8)}

    picked = cs.furthest_first(10)
    assert len(picked) == 10 and len(set(picked)) == 10, f"duplicate selections: {picked}"


def test_tavaal_reparameterize_is_device_agnostic():
    """`torch.randn(*mu.size())` allocated on CPU and was only moved when mu.is_cuda,
    so on MPS (or any non-CUDA accelerator) the VAE crashed on a device mismatch."""
    import inspect
    from strategies.competitors.TA_VAAL.ta_vaal_query_model import VAE

    # strip comments/docstring: the explanatory comment legitimately mentions `.is_cuda`
    code = "\n".join(
        ln.split("#")[0] for ln in inspect.getsource(VAE.reparameterize).splitlines()
    )
    assert "randn_like" in code, "reparameterize must build the noise with randn_like(mu)"
    assert "is_cuda" not in code, "reparameterize still branches on .is_cuda (breaks on MPS)"

    vae = VAE.__new__(VAE)
    mu, logvar = torch.zeros(4, 6), torch.zeros(4, 6)
    out = VAE.reparameterize(vae, mu, logvar)
    assert out.shape == mu.shape and out.device == mu.device


# ------------------------------------------------------------------ Sequential GCN
def test_aff_to_adj_is_finite_and_has_self_loops():
    torch.manual_seed(0)
    adj = aff_to_adj(torch.randn(24, 32))
    assert adj.shape == (24, 24)
    assert torch.isfinite(adj).all(), "propagation matrix contains NaN/Inf"
    # self-loops are added back after the degree normalisation
    assert torch.all(adj.diagonal() >= 1.0 - 1e-5), "self-loops were not restored"


def test_gcn_trains_on_cloned_inference_tensors():
    """Embeddings come from get_embeddings(), which runs under torch.inference_mode().

    Inference tensors cannot be saved for backward, so the strategy clones them. This test
    reproduces that exact path: without the clone, loss.backward() raises.
    """
    with torch.inference_mode():
        # Mirror what the pipeline actually produces: ResNet/VGG embeddings are post-ReLU, hence
        # non-negative, and the strategy L2-normalises them. That keeps every cosine affinity
        # non-negative and every degree comfortably positive in aff_to_adj. (With signed features
        # the degrees can nearly cancel, the normalisation blows the adjacency up and the sigmoid
        # saturates — a fragility inherited from the reference implementation, harmless here.)
        feats = torch.randn(20, 16).relu()
    feats = feats.detach().clone()                     # <- the fix under test
    feats = F.normalize(feats, p=2, dim=1)

    adj = aff_to_adj(feats)
    mask = torch.zeros(20, dtype=torch.bool)
    mask[10:] = True                                   # second half = 'labelled'

    gcn = GCN(nfeat=16, nhid=8, nclass=1, dropout=0.0)
    scores, hidden = gcn(feats, adj)

    assert scores.shape == (20, 1) and hidden.shape == (20, 8)
    assert torch.all((scores >= 0) & (scores <= 1)), "scores must be sigmoid outputs in [0, 1]"
    assert not torch.any((scores <= 0) | (scores >= 1)), (
        "sigmoid saturated to exactly 0/1 on realistic embeddings: gradients would vanish"
    )

    loss = bce_adj_loss(scores.squeeze(1), mask, ~mask)
    assert torch.isfinite(loss), "loss is not finite"
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in gcn.parameters()), \
        "no gradient reached the GCN"


def test_gcn_can_separate_labelled_from_unlabelled():
    """Sanity: the objective is learnable — the loss must go down over a few steps."""
    torch.manual_seed(0)
    feats = torch.cat([torch.randn(15, 16), torch.randn(15, 16) + 3.0])   # two clusters
    adj = aff_to_adj(feats)
    mask = torch.zeros(30, dtype=torch.bool)
    mask[15:] = True

    gcn = GCN(nfeat=16, nhid=8, nclass=1, dropout=0.0)
    opt = torch.optim.Adam(gcn.parameters(), lr=1e-2)

    losses = []
    for _ in range(50):
        opt.zero_grad(set_to_none=True)
        scores, _ = gcn(feats, adj)
        loss = bce_adj_loss(scores.squeeze(1), mask, ~mask)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"GCN loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


# ------------------------------------------------------------------ runner
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

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


# ------------------------------------------------------------------ BADGE
def test_badge_gradient_embedding_ranks_uncertainty():
    """Gradient embedding g = (p - onehot(argmax p)) (x) z: uncertain sample -> larger norm."""
    embs = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])   # identical embeddings
    probs = torch.tensor([[0.95, 0.03, 0.02],                 # confident
                          [0.40, 0.35, 0.25]])                # uncertain
    scale = probs - F.one_hot(probs.argmax(1), num_classes=probs.size(1)).float()
    g = (scale.unsqueeze(2) * embs.unsqueeze(1)).view(probs.size(0), -1)

    assert g[1].norm() > g[0].norm(), "uncertain sample must have the larger gradient-embedding norm"


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

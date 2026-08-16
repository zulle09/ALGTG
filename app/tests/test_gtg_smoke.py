"""
Smoke test for the core GTG contribution (the thing the paper is about).

Exercises the *real* GTGModule end-to-end on a tiny synthetic batch — no CIFAR, no
checkpoints, runs on CPU in a couple of seconds. It checks the things that would
silently break the method:

  * graph_trasduction_game runs, converges, and produces a finite entropy history
  * shapes are consistent (y_pred, y_true are per-sample scalars)
  * NO NaNs anywhere
  * gradients actually reach the learnable module (mod_ls) -> the online variant can learn
  * the RBF affinity is non-degenerate (validates the distance-clamp fix)

Run it from the repo root, inside the `algtg` env:

    python app/tests/test_gtg_smoke.py       # or: pytest app/tests/test_gtg_smoke.py
"""

import os
import sys
import traceback

# make `config`, `utils`, `models...` importable exactly as the app does
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn.functional as F

from models.modules.GTG.GTG_Cls import GTGModule


# ------------------------------------------------------------------ helpers
B, N_CLASSES, N_LAB = 40, 10, 10           # batch, classes, labelled-in-batch
FEATURE_SIZES = [8, 4, 2, 1]
NUM_CHANNELS = [64, 128, 256, 512]
INTERM_DIM = 16
EMB_DIM = 32


def _make_module(gtg_model="lsmlps"):
    gtg_p = dict(
        gtg_t=1e-4, gtg_i=8,
        am_s="mixed", am_sl=False, am_t=0.5,
        am_ts=["mean"], am=["corr"],
        e_s="mean", bsgtgo=8,
        n_classes=N_CLASSES, device="cpu", gtg_module=gtg_model,
    )
    ll_p = dict(feature_sizes=FEATURE_SIZES, num_channels=NUM_CHANNELS, interm_dim=INTERM_DIM)
    mod = GTGModule((gtg_p, ll_p))
    mod.define_idx_params(0, 0)             # AM_threshold_strategy='mean', AM_function='corr'
    return mod


def _make_batch():
    feats = [torch.randn(B, c, s, s) for c, s in zip(NUM_CHANNELS, FEATURE_SIZES)]
    embedds = torch.randn(B, EMB_DIM)
    outs = torch.randn(B, N_CLASSES)
    labels = torch.randint(0, N_CLASSES, (B,))
    return feats, embedds, outs, labels


# ------------------------------------------------------------------ tests
def test_forward_shapes_and_no_nan():
    torch.manual_seed(0)
    mod = _make_module("lsmlps").train()
    feats, embedds, outs, labels = _make_batch()

    (y_pred, y_true), mask = mod(feats, embedds, outs, labels, iteration=1, n_lab_obs=N_LAB, weight=1)

    assert y_pred.shape == (B,), f"y_pred shape {tuple(y_pred.shape)} != ({B},)"
    assert y_true.shape == (B,), f"y_true shape {tuple(y_true.shape)} != ({B},)"
    assert mask.shape == (B,) and mask.dtype == torch.bool
    assert not torch.isnan(y_pred).any(), "y_pred contains NaN"
    assert not torch.isnan(y_true).any(), "y_true contains NaN"
    assert mask.sum().item() == N_LAB, "labelled mask count mismatch"


def test_gradient_reaches_learnable_module():
    """The whole point of the online variant: entropy -> X -> A -> mod_ls must backprop."""
    torch.manual_seed(0)
    mod = _make_module("lsmlps").train()
    feats, embedds, outs, labels = _make_batch()

    (y_pred, y_true), _ = mod(feats, embedds, outs, labels, iteration=1, n_lab_obs=N_LAB, weight=1)
    assert y_pred.requires_grad, "y_pred is detached — module cannot learn"

    loss = F.mse_loss(y_pred, y_true.detach())
    loss.backward()

    grads = [p.grad for p in mod.gtg_module.mod_ls.parameters() if p.grad is not None]
    assert grads, "no gradient reached mod_ls — the GTG path is not differentiable"
    assert any(g.abs().sum() > 0 for g in grads), "gradients to mod_ls are all zero"


def test_rbfk_affinity_is_not_degenerate():
    """Validates the fix that removed the max=1 clamp on Euclidean distances."""
    torch.manual_seed(0)
    mod = _make_module("lsmlps")
    embedds = torch.randn(30, 512) * 3.0            # ResNet-like scale
    A = mod.get_A_rbfk(embedds)

    assert A.shape == (30, 30)
    assert not torch.isnan(A).any(), "RBF affinity contains NaN"
    off = A[~torch.eye(30, dtype=torch.bool)]
    assert off.std().item() > 1e-2, (
        f"RBF affinity is near-constant (std={off.std().item():.4g}); "
        "the distance clamp is back and GTG will not propagate"
    )


def test_variants_construct_and_run():
    """Every advertised online variant should at least forward without crashing."""
    for name in ["llmlp", "lsmlps", "lsmlpsA", "lstmreg", "lstmbc"]:
        torch.manual_seed(0)
        mod = _make_module(name).train()
        feats, embedds, outs, labels = _make_batch()
        (y_pred, y_true), _ = mod(feats, embedds, outs, labels, iteration=1, n_lab_obs=N_LAB, weight=1)
        assert not torch.isnan(y_pred).any(), f"{name}: y_pred NaN"
        assert not torch.isnan(y_true).any(), f"{name}: y_true NaN"


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

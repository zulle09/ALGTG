"""
Regression tests for the theoretical properties ALGTG claims in the paper.

Unlike a reimplementation, these drive the *real* GTGModule: the payoff matrix is
injected directly (by patching `get_A`) so the actual replicator loop, the labelled
clamping and the entropy-history construction are all exercised.

Covered:
  * Proposition 1  — with labelled players clamped, W_LL and W_LU are inert, and only
                     W_UU / W_UL affect the acquisition scores (incl. a sensitivity
                     control, so the test cannot pass vacuously)
  * redundancy term (`am_r`) — scales the UL/LU blocks, and monotonically widens the
                     score gap between redundant and informative samples
  * isolated nodes — stay on the simplex instead of collapsing to the zero vector
  * `diversity`    — `1 - A` must not silently re-introduce a maximal self-loop
  * `corr`         — a constant (zero-variance) embedding row must not yield NaN

    python app/tests/test_gtg_theory.py      # or: pytest app/tests/test_gtg_theory.py
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from models.modules.GTG.GTG_Cls import GTGModule


N_LAB, N_TOT, N_CLASSES, T_ITERS = 2, 6, 2, 25


# ------------------------------------------------------------------ helpers
def _make_module(n_classes=N_CLASSES, am_s="uncertanity", am_r=0.0, am_sl=False,
                 am_ts="none", am="corr", gtg_i=T_ITERS, gtg_t=1e-6):
    gtg_p = dict(
        gtg_t=gtg_t, gtg_i=gtg_i,
        am_s=am_s, am_sl=am_sl, am_t=0.5,
        am_ts=[am_ts], am=[am], am_r=am_r,
        e_s="mean", bsgtgo=8,
        n_classes=n_classes, device="cpu", gtg_module="lsmlps",
    )
    ll_p = dict(feature_sizes=[8, 4, 2, 1], num_channels=[64, 128, 256, 512], interm_dim=16)
    mod = GTGModule((gtg_p, ll_p))
    mod.define_idx_params(0, 0)
    return mod


def _base_payoff():
    """Symmetric, non-negative, zero diagonal. Rows 0-1 labelled, 2-5 unlabelled."""
    return torch.tensor([
        [0.00, 0.80, 0.70, 0.20, 0.10, 0.30],
        [0.80, 0.00, 0.20, 0.90, 0.30, 0.10],
        [0.70, 0.20, 0.00, 0.40, 0.60, 0.50],
        [0.20, 0.90, 0.40, 0.00, 0.50, 0.70],
        [0.10, 0.30, 0.60, 0.50, 0.00, 0.40],
        [0.30, 0.10, 0.50, 0.70, 0.40, 0.00],
    ], dtype=torch.float32)


def _scores(A, n_lab=N_LAB, n_tot=N_TOT, gtg_i=T_ITERS, gtg_t=1e-6):
    """Run the REAL GTG dynamics with payoff A; return the per-sample acquisition scores.

    NOTE: `entropy_hist` is zero-padded when the loop exits early, so the mean depends on how
    many iterations actually ran. Callers that compare scores *across configurations* must pin
    `gtg_i` and use `gtg_t=0.` so every run executes the same number of iterations.
    """
    mod = _make_module(gtg_i=gtg_i, gtg_t=gtg_t)
    labels = torch.tensor([0, 1] + [0] * (n_tot - n_lab))
    mod.batch_size = n_tot
    mod.n_lab_obs = n_lab
    mod.lab_labels = labels[:n_lab]
    mod.get_X(torch.zeros(n_tot, N_CLASSES))
    mod.get_A = lambda _emb, _A=A: _A                       # inject the payoff directly
    _, ent_hist = mod.graph_trasduction_game(torch.zeros(n_tot, 4), labels)
    return ent_hist[n_lab:].mean(dim=1)                     # 'mean' entropy-trajectory score


def _invert_block(A, rows, cols):
    B = A.clone()
    B[rows, cols] = 1 - B[rows, cols]
    return B


# ------------------------------------------------------------------ Proposition 1
def test_proposition1_LL_and_LU_blocks_are_inert():
    """Clamped labelled players => W_LL and W_LU cannot influence the acquisition."""
    L, U = slice(0, N_LAB), slice(N_LAB, N_TOT)
    A = _base_payoff()
    base = _scores(A)

    for name, rows, cols in [("W_LL", L, L), ("W_LU", L, U)]:
        got = _scores(_invert_block(A, rows, cols))
        assert torch.allclose(base, got, atol=1e-6), (
            f"{name} changed the scores by {(base - got).abs().max():.3e} — "
            "Proposition 1 is violated (are labelled players still clamped?)"
        )


def test_proposition1_is_not_vacuous():
    """Control: the blocks that DO drive the game must change the scores."""
    L, U = slice(0, N_LAB), slice(N_LAB, N_TOT)
    A = _base_payoff()
    base = _scores(A)

    for name, rows, cols in [("W_UU", U, U), ("W_UL", U, L)]:
        got = _scores(_invert_block(A, rows, cols))
        assert not torch.allclose(base, got, atol=1e-6), (
            f"{name} left the scores unchanged — the test harness is insensitive "
            "and the inertness tests above would pass vacuously"
        )


# ------------------------------------------------------------------ redundancy term
def test_redundancy_term_scales_only_UL_and_LU():
    """`am_r` must scale the labelled<->unlabelled coupling and nothing else."""
    lam = 2.0
    emb = torch.randn(N_TOT, 16)

    mod0, modr = _make_module(am_r=0.0), _make_module(am_r=lam)
    for m in (mod0, modr):
        m.n_lab_obs = N_LAB
    A0, Ar = mod0.get_A(emb), modr.get_A(emb)

    L, U = slice(0, N_LAB), slice(N_LAB, N_TOT)
    assert torch.allclose(Ar[U, L], A0[U, L] * (1 + lam), atol=1e-5), "UL block not scaled"
    assert torch.allclose(Ar[L, U], A0[L, U] * (1 + lam), atol=1e-5), "LU block not scaled (symmetry)"
    assert torch.allclose(Ar[U, U], A0[U, U], atol=1e-6), "UU block must be untouched"
    assert torch.allclose(Ar[L, L], A0[L, L], atol=1e-6), "LL block must be untouched"


def test_redundancy_term_widens_redundant_informative_gap():
    """Increasing lambda must push redundant samples down relative to informative ones."""
    #        lab0  lab1   redundant(~lab0)  informative(between classes)
    A = torch.tensor([
        [0.00, 0.10, 0.95, 0.50],
        [0.10, 0.00, 0.05, 0.50],
        [0.95, 0.05, 0.00, 0.30],
        [0.50, 0.50, 0.30, 0.00],
    ], dtype=torch.float32)

    gaps = []
    for lam in (0.0, 1.0, 4.0):
        B = A.clone()
        B[N_LAB:, :N_LAB] *= (1 + lam)
        B[:N_LAB, N_LAB:] *= (1 + lam)
        # pin the iteration count (gtg_t=0.) so zero-padding is identical across lambdas and
        # the three means stay directly comparable
        s = _scores(B, n_lab=N_LAB, n_tot=4, gtg_i=6, gtg_t=0.)
        gaps.append((s[1] - s[0]).item())            # informative - redundant

    assert all(g > 0 for g in gaps), f"redundant sample never scored below the informative one: {gaps}"
    assert gaps[0] < gaps[1] < gaps[2], f"gap is not monotone in lambda: {gaps}"


# ------------------------------------------------------------------ pathologies
def test_isolated_node_stays_on_the_simplex():
    """A node with no payoff must keep its strategy, not collapse to the zero vector."""
    A = _base_payoff()
    A[5, :] = 0.
    A[:, 5] = 0.                                     # node 5 fully isolated

    mod = _make_module()
    labels = torch.tensor([0, 1, 0, 0, 0, 0])
    mod.batch_size, mod.n_lab_obs, mod.lab_labels = N_TOT, N_LAB, labels[:N_LAB]
    mod.get_X(torch.zeros(N_TOT, N_CLASSES))
    mod.get_A = lambda _emb, _A=A: _A
    Xs, ent_hist = mod.graph_trasduction_game(torch.zeros(N_TOT, 4), labels)

    assert torch.isclose(Xs[5, :, 0].sum(), torch.tensor(1.0), atol=1e-5), (
        f"isolated node left the simplex (row sums to {Xs[5, :, 0].sum():.3f})"
    )
    # Assert on the first recorded iteration rather than the mean: the mean is diluted by the
    # zero-padding applied when the loop exits early, which would make the threshold arbitrary.
    ent_iso = ent_hist[5, 0]
    assert ent_iso > 0.9, (
        f"isolated node has entropy {ent_iso:.4f} at t=0 (expected ~1.0 for {N_CLASSES} "
        "classes): it collapsed toward zero entropy and could never be queried"
    )


def test_diversity_does_not_reintroduce_self_loops():
    """`A = 1 - A` turns a zero diagonal into 1; the convention must be restored."""
    mod = _make_module(am_s="diversity", am_sl=False)
    mod.n_lab_obs = N_LAB
    A = mod.get_A(torch.randn(N_TOT, 16))
    assert torch.allclose(A.diagonal(), torch.zeros(N_TOT), atol=1e-6), (
        f"diversity re-introduced self-loops: diagonal = {A.diagonal().tolist()}"
    )


def test_corr_affinity_handles_constant_row():
    """A dead/constant embedding row has undefined correlation -> must map to 0, not NaN."""
    mod = _make_module(am="corr")
    emb = torch.randn(N_TOT, 16)
    emb[3] = 0.                                      # zero-variance row
    A = mod.get_A_corr(emb)
    assert not torch.isnan(A).any(), "corr affinity still produces NaN on a constant row"


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

# ALGTG — paper (arXiv preprint)

Draft scaffold for the ALGTG preprint. Generic LaTeX — **no conference style file needed**, compiles with a stock TeX Live / MacTeX install.

## Files
- `main.tex` — paper. **Related Work, Background and Method are drafted**; everything marked `%% TODO` needs real numbers/figures.
- `references.bib` — real citations (verify venues/years before posting).

## Build
```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Produces `main.pdf`.

## Posting to arXiv
- Upload the **source**, not just the PDF: `main.tex`, `references.bib`, any figure files, **and the generated `main.bbl`** (arXiv is most reliable when you include the `.bbl` so it doesn't have to run BibTeX itself).
- Fill in the author/affiliation block in `main.tex` (currently a placeholder).
- Pick the primary category **cs.LG** (secondary: cs.CV) when submitting.
- No page limit on arXiv, but keep it tight; you can always submit to a venue later.

## Before you write a single number
1. **Re-run every experiment with the fixed baselines.** BALD, AlphaMix and BADGE were broken/unfaithful before this session; any pre-fix result is invalid. Sanity-check first:
   ```
   python app/tests/test_baseline_fixes.py
   python app/tests/test_gtg_smoke.py
   python app/tests/test_gtg_theory.py
   ```
   `test_gtg_theory.py` guards the claims the paper makes: Proposition 1 (the W_LL / W_LU
   blocks are inert), the redundancy term's effect, and the pathologies that were fixed.
   If one of these fails, a statement in the paper is no longer true of the code.
2. **Pick ONE flagship ALGTG variant** (best/most principled) and demote the other six to an architecture ablation.
3. **Add Sequential GCN** (Caramalau et al., CVPR 2021) — the closest graph-based competitor.

## Ablations that must be in the paper
- **`-am_r` sweep** (λ = 0, 0.5, 1, 2, …): λ=0 is standard GTG. Quantifies how much explicit
  redundancy amplification adds over the *emergent* behaviour. This replaces the old
  "mixed payoff on/off" ablation, which was vacuous (see Proposition 1: the LL/LU blocks
  are inert, so `am_md=LL|LU` has no effect).
- **Acquisition-signal control**: full entropy *trajectory* vs. entropy at convergence only
  vs. plain softmax entropy. This is the central claim — without it a reviewer cannot tell
  whether the dynamics matter at all.

## Checklist
- [ ] Mean ± CI over ≥5 trials on every curve (already supported by `-tr`).
- [ ] Ablation isolating the *mixed* payoff term (your core claim).
- [ ] Report compute/runtime honestly (affinity + GTG scale with pool size).
- [ ] Author/affiliation filled in; correspondence email correct.

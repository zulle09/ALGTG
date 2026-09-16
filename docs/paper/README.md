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
   python app/tests/test_utils_fixes.py
   ```
   `test_gtg_theory.py` guards the claims the paper makes: Proposition 1 (the W_LL / W_LU
   blocks are inert), the redundancy term's effect, and the pathologies that were fixed.
   If one of these fails, a statement in the paper is no longer true of the code.
   `test_utils_fixes.py` guards the reporting path — CSV integrity, CI computation with a
   single trial, figure leaks — i.e. the code that produces the numbers you will publish.
2. **Pick ONE flagship ALGTG variant** (best/most principled) and demote the other six to an architecture ablation.
3. ~~Add Sequential GCN~~ — **done**: `uncertaingcn` and `coregcn` are implemented
   (`app/strategies/competitors/GCN/`) and registered in the CLI.

## Running the benchmark

Diagnostics (t-SNE, affinity plots, image dumps) are **off by default** — they can cost more
wall-clock than the training itself. Add `-diag` on a single run if you want the figures.

Smoke run first (catches crashes before a multi-day sweep):
```
python app/main.py -gpus 1 -tr 1 -ds cifar10 -m random entropy coreset badge bald cdal alphamix uncertaingcn coregcn ll gtg
```
Then the full sweep via `run_sweep.py`, which runs **one method per process** into a shared
results folder, records failures without aborting the rest, and **skips whatever already
finished** when re-run — so a crash or an SSH drop costs one method, not the whole sweep:

```
python run_sweep.py -o run1 -ds cifar10 --trials 1 -m random     # time one method first
python run_sweep.py -o run1 -ds cifar10 --trials 5               # the real sweep
python run_sweep.py -o run1 -ds cifar10 --status                 # progress at any time
```
Launch it under `tmux`/`nohup`. It prints a running ETA and, at the end, the exact
`make_results.py` command to run.

Per-method isolated smoke check (miniature protocol, seconds per method — use this to catch
crashes before a multi-day run):
```
for m in random entropy coreset badge bald cdal alphamix uncertaingcn coregcn ll tidal tavaal gtg ll_gtg lsmlps_gtg; do python app/tests/smoke_run.py $m || echo "FAIL $m"; done
```

## From results to paper

`make_results.py` turns the benchmark CSV into the table and figure, so no number is ever
transcribed by hand:
```
python docs/paper/make_results.py results/<exp_path> cifar10 --trials 5 --budgets 2000 4000 6000 8000 10000
```
`--trials` is required: the script refuses to build anything when a strategy/budget cell doesn't
have exactly that many rows, when rows are duplicated, or when one table row would pool several
strategies, since each of those narrows the reported CI.
It writes `results_cifar10.tex` (booktabs table, best per column in bold), `curve_cifar10.pdf`
(accuracy vs. budget with CI bands) and `summary_cifar10.txt` (AUBC ranking, for the prose).
`main.tex` picks the table up automatically via `\IfFileExists`, falling back to a placeholder
when it is absent — so the document always compiles. Missing method/budget cells render as `--`
rather than being invented.

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

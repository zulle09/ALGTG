# Analysis plan — what to check in the data, and what to write

Written before the numbers exist, deliberately: deciding in advance what would confirm or
refute each claim is what stops the Results section from becoming a post-hoc rationalisation
of whatever the runs happen to show. Fill in, do not rewrite, the claims.

---

## 1. Sanity checks — run these before reading anything into the results

If any of these fails, the numbers are not yet interpretable and no prose should be written.

| Check | Expected | If it fails |
|---|---|---|
| Random is the weakest, or close to it | yes, on every dataset | a method that loses to Random has a bug, not a finding |
| BALD > Random | clearly | the MC-dropout fix did not take effect |
| BADGE ≳ Core-Set | roughly comparable | gradient embeddings not being used |
| Curves increase monotonically with budget | yes | training instability, not acquisition quality |
| CI width shrinks with more trials | yes | seeds are not actually varying |
| All methods have the same number of rows in `results.csv` | yes | a method crashed mid-sweep |

Cheap way to see all of this at once: `summary_<ds>.txt` from `make_results.py` (AUBC ranking).

---

## 2. The claims, and what would support each

Write each of these only if the data supports it. Where it does not, say so — a paper that
reports an honest partial win is far more credible than one that overclaims.

**C1 — ALGTG is competitive with or better than the state of the art.**
Support: highest or joint-highest AUBC on a majority of datasets, with non-overlapping CIs at
the budgets where it leads. *If it wins on some datasets and not others*, that is the finding:
characterise where (many classes? low budget?) rather than burying it.

**C2 — the entropy *trajectory* is what carries the signal, not the final labelling.**
This is the central claim and the one a reviewer will target hardest.
Support: ablation (v) — trajectory vs. entropy-at-convergence vs. plain softmax entropy, all
else identical. **If trajectory ≈ final entropy, the paper's thesis is weakened and must be
reframed** around what does work (e.g. the transductive graph itself). Decide this before
writing the abstract.

**C3 — redundancy avoidance is emergent (Proposition 1).**
Support: the λ sweep. λ=0 already outperforms pure-uncertainty baselines → emergence.
λ>0 improving further → the term is a useful knob. λ>0 *not* helping is a perfectly
publishable result: report it, and keep λ=0 as the default.

**C4 — the online variant is worth its cost.**
Honest framing: it is trained to regress the offline signal, which is available at query time
anyway. It earns its place only if it beats the offline variant at equal budget — because the
auxiliary loss shapes the backbone. *If it does not*, say so and present offline ALGTG as the
method, online as an ablation. Do not let a reviewer discover this.

---

## 3. Objections to preempt in the Discussion

1. **"Cost."** Quadratic in the pool. Report wall-clock per query next to accuracy. If ALGTG is
   much slower than Core-Set for a small gain, address it head-on rather than omitting timings.
2. **"Why not just Sequential GCN?"** Both build a graph over labelled+unlabelled. The
   distinction is learned embeddings vs. dynamics of a fixed-payoff game. Needs an empirical
   difference, not only a conceptual one.
3. **"Sensitivity to the affinity kernel."** If results swing wildly across cos/corr/RBF, the
   method is fragile. Report the spread; do not show only the best kernel.
4. **"Sub-sampled pool."** 10k of the pool per cycle. Note it is applied identically to every
   baseline, so it cannot favour ALGTG.
5. **"One backbone family."** ResNet-18 everywhere except Caltech-256. State it as a limitation.

---

## 4. Prose structure, once the numbers are in

1. *Main results* — one paragraph per dataset group, referencing `tab:main-<ds>` and the curves.
   Lead with AUBC, then the budgets where the gap is largest. Name the strongest competitor
   explicitly rather than "prior work".
2. *Ablations* — one short paragraph each, in the order (i)–(v) already in `main.tex`.
   Ablation (v) deserves the most space: it is C2.
3. *Analysis* — entropy-trajectory plots for queried vs. non-queried samples (the mechanism
   figure), and the cost discussion.
4. *Back-fill*: the abstract's headline sentence, the Introduction's closing claim, and the
   Conclusion's first `%% TODO` — all three must state the **same** result, in the same terms.

---

## 5. Order of operations

```
run_sweep.py  ->  summary_<ds>.txt (§1 sanity)  ->  §2 decide which claims hold
              ->  make_results.py (table + curves)  ->  write §4  ->  arXiv
```

Do not write prose before §1 passes.

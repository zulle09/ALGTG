# Decisions

Append-only log, newest first. Add an entry when a choice with real alternatives is settled; skip
routine work. "(inferred — confirm)" marks rationale reconstructed from code or commits rather than
found stated. Verify or delete those.

Format:

    ## YYYY-MM-DD — <short title>
    <what we chose and why>
    <what would make us revisit>

---

## 2026-10-16 — BAIT is implemented from the paper, not translated from dal-toolbox (inferred — confirm)
dal-toolbox's `bait.py` has no attribution but mirrors the licence-less `JordanAsh/badge` code identifier by
identifier and comment by comment, so a translation (with dal-toolbox's BSD notice) would carry that provenance
and BSD cannot launder it. The alternatives were to translate anyway, to translate the badge file (no licence at
all), or to skip BAIT. The port is written from the paper's equations with a different formulation (Kronecker-
structured factors, rank-r updates of the per-candidate scores), and both reference files are used only as
numerical oracles in tests. dal-toolbox's version also has a scale bug (one Fisher mean per batch of 32, which
inflates the labelled prior by ceil(L / 32)) that the badge code does not.
Revisit if the dal-toolbox authors or the BADGE author confirm a licence that covers the code.

## 2026-10-16 — BAIT's exact Fisher runs only where it fits; the approximation is a separate, named variant
The softmax Fisher has dimension C*D: the inverse is 210 MB at C = 10 and 21 GB at C = 100, and the reference
embeddings alone are 205 GB. `fisher: 'ce'` is refused at construction above `max_embedding_dim` (12,000), which
leaves CIFAR-10, SVHN and Fashion-MNIST, with the way out in the message. `fisher: 'bce'` (dal-toolbox's Bernoulli
likelihood on the top probability, dimension D for any C) runs everywhere and is named `ResNet_BAIT_bce`. The
alternatives were a silent fall-back (changes the algorithm without saying so in the results), subsampling the
classes or the pool (the paper does neither), or skipping BAIT on the other datasets.
Revisit when the exact variant is needed for 100+ classes; that needs a GPU with tens of GB and a Kronecker-
factored inverse, not this formulation.

## 2026-10-16 — ProbCover's radius is the paper's purity rule, re-estimated on every query
`delta: None` in `competitor_config.py`. The alternatives were a fixed radius, dal-toolbox's default of 0.6 or the
authors' 0.5-0.6 per dataset, which belong to L2-normalised SimCLR features and mean nothing on a supervised
embedding that changes every cycle, and dal-toolbox's `estimate_delta`, which is never called and is wrong. The
paper's criterion (largest radius with 95 % pure balls against k-means pseudo-labels, k = number of classes) is
solved exactly as an order statistic, not on a grid. A fixed radius stays available and is part of the strategy
name (`..._d0.55`), as is a different alpha (`..._a0.9`).
Revisit if real embeddings saturate (the labelled set covering every candidate turns the whole query into the
random tail), or if self-supervised features become the main setting and the authors' fixed values are wanted.

## 2026-10-16 — When ProbCover has nothing left to cover, the rest of the query is seeded random (inferred — confirm)
Both reference codes repeat index 0 then: a duplicate and, whenever there is a labelled sample, a labelled one,
which `update_sets` rejects, so something else had to be chosen and the paper does not say what. Random draws from
the unselected candidates are the least inventive option; the alternatives were the highest static degree first
(redundant, dense regions) or covering the pool again with the selected points excluded (a procedure the paper does
not have). The number of random picks is logged on every query.
Revisit if saturation is common on real features: then a method designed for all budgets (UHerding, MaxHerding) is
the better comparison, or the fallback needs a principled replacement.

## 2026-10-16 — TypiClust defaults to the authors' selection rule, not dal-toolbox's (inferred — confirm)
The brief names dal-toolbox as the source, and its `TypiClust` is ported too (`selection: 'dal'`, golden-tested
against the original). The default is the rule of the authors' code: dal-toolbox queries the largest cluster
that holds no labelled point and picks uniformly at random once none is left, and with K capped at 500 and
B = 1000 that was 72-87 % random picks on synthetic 512-d features (L = 1000...9000), a Random baseline under
another name. The authors' rule sorts clusters by fewest labelled points, then size, and cycles through them.
Each variant has its own strategy name.
Revisit if the main table should show the literal dal-toolbox behaviour, or if real features show the random
fallback is rare.

## 2026-10-16 — Competitor features: the model's embedding by default, a `.npy` for self-supervised ones
`-fs model|file` (+ `-fp`). `model` is the penultimate embedding of the current cycle's model, what CoreSet and
BADGE use (the default the user asked for). `file` reads precomputed features; no SimCLR/SCAN training code is
copied into the repo (user constraint), so the user produces the file. A strategy on a file is named `..._ssl`:
rows from different sources must not merge in `results.csv`, and a file is representation learning on the whole
unlabelled set, which the baselines do not get, so those rows should be reported as a separate group (suffix and
grouping: inferred — confirm). The file path is not part of the name: use one `-o` per feature file.
Revisit when real SimCLR features exist and the main table has to pick a group.

## 2026-10-16 — scikit-learn work in competitors runs under `threadpool_limits(1)` (workaround)
In the first smoke run TypiClust aborted (exit 139, `OMP: Error #179 pthread_mutex_init failed`) inside
scikit-learn's `euclidean_distances`, a BLAS call, after torch had trained on MPS; with the limit both smoke
runs pass. The same calls pass in isolation (MPS, DataLoader workers and `OMP_NUM_THREADS` set after the torch
import, each tried), so which ingredient triggers it is not isolated; the duplicate OpenMP runtimes behind
`KMP_DUPLICATE_LIB_OK` are the likely cause. One thread also matches the benchmark's intent
(`OMP_NUM_THREADS=1`) and makes clustering reproducible.
Revisit if the environment is fixed or the GPU host never shows it; then the limit only costs speed.

## 2026-09-16 — GTG's NaN-hunting logs move to DEBUG behind `--debug_gtg`
The gradient hooks and per-replicator-iteration state logs in `GTG_Cls.py` (mirrored in
`GTG_off.py`) were added to chase weights diverging to NaN, and they work — but each formats a
tensor and synchronises with the GPU: ~11 KB and ~136 syncs per training batch, ~3-4 GB of log per
online method over a CIFAR-10 sweep. They are now DEBUG, raised only by `main.py --debug_gtg`
(measured on the smoke protocol: 188 KB -> 28 KB with the flag off, identical output with it on).
Deleting them was the alternative and was rejected: they are the only trace of *where* a NaN
enters. The NaN assertions on `y_pred`/`y_true`/embeddings stay at all times, so a run still fails
loudly. The same commit logs `git_commit` at run start and stores it in `yamal_conf.yamal`, so a
number can be traced back to the code that produced it.
Revisit if a NaN turns out to need more than one batch of gradient history.

## 2026-09-16 — Test accuracy is sample-weighted, and CIs use Student-t
`Cls_TrainWorker.test` averaged per-batch accuracies, which over-weights the short last batch: on
CIFAR-10 a constant-class predictor (true accuracy 0.1000) reported up to 0.1010 at batch size 128
and 0.1021 at 256 — and online GTG uses 256 while the baselines use 128, so the error differed
between method families. It now sums correct predictions over the test set. In the same spirit,
`make_results.py` and `plot_res_std_mean` use t(0.975, n-1) instead of z = 1.96, which understated
a 5-trial CI by ~30%. Both change reported numbers, so results produced earlier are not comparable.
Revisit only if the metric definition changes; train-curve accuracy is still a per-batch mean
(curves only, never reported).

## 2026-09-16 — TiDAL's moving average lives in shared memory
`moving_prob` is updated in the main process but read through DataLoader workers. With `fork` (the
Linux default) the workers keep a private copy, so with the default `-nw 4` TiDAL's KL target was
always zeros; macOS `spawn` hid this, so no smoke run caught it. It is now allocated with
`share_memory_()` (`test_pipeline_fixes.py` locks it in with a fork read-back). Any TiDAL number
produced on Linux with workers is invalid.
Revisit if batches ever stop being unique per epoch, since prefetching would then race.

## 2026-09-16 — Trial-level resume instead of restarting a crashed method
`main.py -ex/--existing {append,resume,overwrite}` decides what happens to rows already in
`results.csv` for a strategy and trial, and `run_sweep.py` records each method's `--trials` and
`--extra` in `.sweep_<ds>_<method>.json` and re-launches a crashed method with `resume`. Before
this, resuming appended a second copy of every finished trial, and a `--trials 1` timing run into
the same `-o` marked the method done so the real sweep skipped it. Restarting the method from trial
0 would have been simpler, but on a preemptible GPU it throws away finished trials. Trials are
independent (splits and pools are seeded by dataset/trial/cycle, every cycle reloads `_init` and
re-seeds), which is what makes skipping them sound.
Revisit if trials stop being independent, or if a scheduler takes over retries.

## 2026-09-16 — make_results refuses data that would narrow the CI
`validate()` exits on duplicated rows, on any strategy/budget cell without exactly `--trials` rows,
and when one display row would pool several strategies; `ll_gtg` gets its own "ALGTG-LL (offline)"
row. Pooling different methods or configurations inflates `count` and shrinks the interval, which
is the kind of error a reader cannot see. This supersedes the pooling noted in the 2026-08-17
"Paper numbers are generated" entry.
Revisit to report per-configuration rows (an `-am_r` ablation table), which means extending
`DISPLAY` rather than relaxing the check.

## 2026-09-16 — A failing plot no longer kills a run
`utils.save_figure` logs a `savefig` exception and continues. A broken font stack (matplotlib 3.11
with freetype 2.14 raises "raster overflow" on any text) otherwise killed every method right after
its first AL cycle, because train curves are written per cycle. This is a workaround: plots are
diagnostics and the CSVs carry the numbers, but it is not a fix for the environment, so the runbook
pre-flight now checks that this host can render a figure at all. Only the save is guarded — bugs in
preparing the plotted data still raise.
Revisit if plots ever become load-bearing for a result.

## 2026-09-16 — `-am_ts` defaults to `mean`, and online GTG refuses several configs
The default was `mean none`, so every GTG method ran twice per sweep, and because online variants
share one `GTGModule` whose config is set at construction, both instances trained with the last
combination while their names advertised their own. The default is now the single value `mean` (what
every recorded command already passed) and `get_args()` rejects an online GTG method whose
`-am` × `-am_ts` combinations multiply out to more than one.
Revisit if the online strategies ever get their own module instance.

## 2026-09-16 — No query in the last AL cycle
`ActiveLearner.run` stops after the last cycle's training instead of querying samples that are
never trained on. This saves a full query per trial (for TA-VAAL ~20k VAE iterations, for `GTG_off`
a GTG over ~20k nodes) and removes the trailing `count_classes.csv` row, which now has
`al_iters - 1` rows per trial.
Revisit if an analysis needs the class distribution of a query at the final budget.

## 2026-08-17 — One process per method for benchmark sweeps
`run_sweep.py` runs each method as its own `main.py` process into a shared `-o`. `-o` was added to
`main.py` in the same commit so the processes accumulate into one `results.csv`. The script records
failures and carries on, and skips methods that already have a `.done_` marker. With a single
invocation for all methods, one OOM, preemption or strategy bug kills the rest and wastes paid time
on a rented GPU (`run_sweep.py` docstring).
Revisit if runs move to a scheduler with its own retries, or if `results.csv` gains a key so
re-runs can replace rows instead of appending.

## 2026-08-17 — Expensive diagnostics are opt-in (`-diag`)
t-SNE, affinity-matrix and entropy-history plots and per-sample image dumps are off by default;
accuracy CSVs and train/val curves are always written. On a full benchmark the diagnostics can cost
more wall-clock than training (`-diag` help, `ActiveLearner.__init__` comment). `GTG_off` runs
t-SNE twice on an N×N matrix with N up to ~20k.
Revisit if the paper's mechanism figures need a full sweep's artefacts; even then, prefer one
dedicated `-diag` run over turning it on globally.

## 2026-08-17 — DataLoader workers default to 4 on the single-GPU path
`-nw` defaults to 4 instead of the previous 0. With 0 workers, decoding and augmentation run in the
main process and the GPU sits idle, which is paid idle time on a rented instance (comment in
`Train_DDP.train`). `-nw 0` stays available for containers where workers misbehave.
Revisit on hosts where worker processes hang, or to match the core count.

## 2026-08-17 — Add Sequential GCN as a competitor
`uncertaingcn` and `coregcn` were added because they also build a graph over labelled + unlabelled
data, and "why not just Sequential GCN?" is an expected reviewer objection. The paper has to show
the difference empirically: learned GCN embeddings vs. fixed-payoff game dynamics
(`docs/paper/RESULTS_PLAN.md` §3, `docs/paper/README.md`).
Revisit the paper's framing (not the baseline) if ALGTG and the GCN methods turn out
indistinguishable.

## 2026-08-17 — Paper numbers are generated, never transcribed
`docs/paper/make_results.py` builds the table, curves and AUBC summary from `results.csv`. Missing
method/budget cells render as `--` instead of being filled in. `main.tex` includes the table via
`\IfFileExists`, so the paper always compiles.
Revisit if the table needs per-configuration rows: the substring mapping currently pools all GTG
configurations (see `docs/architecture.md`).

## 2026-08-17 — Analysis plan fixed before results exist
`docs/paper/RESULTS_PLAN.md` sets the sanity checks, and what would confirm or refute claims C1–C4,
before any results exist, so the Results section can't become a post-hoc rationalisation.
Pre-committed outcomes:
- If the entropy trajectory ≈ entropy at convergence (C2), reframe the thesis.
- If online GTG doesn't beat offline at equal budget (C4), offline becomes the method and online
  an ablation.

Revisit by filling the plan in, not by rewriting the claims (the plan's own instruction).

## 2026-08-17 — `lsmlps_gtg` as the flagship online variant (inferred — confirm)
`docs/paper/README.md` says to pick one flagship ALGTG variant and demote the rest to an architecture
ablation. Among the online variants, `run_sweep.py`'s default method list includes only
`lsmlps_gtg`, which suggests that was the pick. No written rationale found.
Revisit once the C4 offline-vs-online comparison is in.

## 2026-08-16 — Discard all pre-fix baseline results
BALD, AlphaMix and BADGE were broken or unfaithful. For example, ResNet18 has no dropout layer, so
BALD's MC-dropout passes were identical; it now applies explicit dropout to the penultimate
embeddings. Every result produced before the fixes is invalid, and all experiments get re-run
(`docs/paper/README.md`). `test_baseline_fixes.py` locks the fixes in.
Revisit if a fix turns out unfaithful to its original method; that method's results are
invalidated again.

## 2026-08-16 — Redundancy term `-am_r` replaces the mixed-payoff ablation
Proposition 1: labelled players are clamped every iteration, so the LL and LU payoff blocks are
inert. `-am_s mixed` and `-am_md LL|LU` change nothing, and an on/off ablation of them would be
vacuous. The real knob is the redundancy weight λ (`-am_r`, scales UL/LU, 0 = standard GTG), and a
λ sweep is a required ablation. `mixed`/`-am_md` stay for backward compatibility.
`test_gtg_theory.py` guards the proposition, including a check that the test can't pass vacuously.
Revisit if labelled clamping changes, because the proposition would stop holding. If λ > 0 doesn't
help, keep λ = 0 and report that (`docs/paper/RESULTS_PLAN.md` C3).

## 2026-07-13 — Assert online-GTG loader alignment instead of truncating
The online worker zips a labelled loader with an unlabelled one. When their lengths differed, `zip`
silently dropped batches, breaking `n_lab_obs` and the per-epoch averages. Training now asserts
equal lengths, which constrains `al_params` and `-bsgtgo` (comment in `Worker.__init__`).
Revisit if online batch composition is redesigned, e.g. with a sampler that guarantees each batch's
labelled share.

## 2026-04-16 — Pin Python 3.11
Downgraded from 3.14 because of incompatibilities with PyTorch and scikit-learn (commit `5e06c11`,
which also notes "TODO: check stability").
Revisit when both support a newer Python.

## 2026-03-28 — Drop the detection task (inferred — confirm)
The SSD/VOC detection pipeline was deleted to focus on image classification (commit `697f95b`). No
reason is recorded. `det_datasets`/`voc_config` in `config.py` are leftovers.
Revisit if detection experiments come back.

## 2024-07-29 — Remove labelled-set oversampling in online GTG (inferred — confirm)
PR #12 removed oversampling of the labelled set in online batches, and with it the CE-loss scaling
by the unlabelled/labelled batch ratio (comment in `Worker.compute_losses`). The commits say "to
test it" and record no outcome.
Revisit if online GTG under-fits the labelled set.

## 2024-07-28 — No self-loops in the affinity matrix by default (inferred — confirm)
The diagonal is 0 unless `-am_sl` is passed (commits `b809991`, `d6161da`). No rationale is
recorded.
Revisit if a self-loop ablation shows an effect.

## 2024-04-21 — Retrain from scratch every AL cycle
Each cycle reloads the model's `_init` checkpoint instead of fine-tuning the previous cycle's
weights, "different from what LL4AL have done" (comment in `Worker.__init__`, commit `53ea4d7`).
Why: no strategy's result then depends on its own earlier training trajectory (inferred — confirm).
Revisit if per-cycle training cost becomes the bottleneck.

## 2024-02-18 — Score a random unlabelled pool each cycle, not the whole unlabelled set
Each cycle queries from `unlab_sample_dim` (10k) random unlabelled samples (commit `b0d60ea`). The
same sub-sampling applies to every baseline, so it can't favour ALGTG (`docs/paper/RESULTS_PLAN.md` §3).
Why: affinity construction and GTG are quadratic in pool size, and the 1k/10k/1k protocol follows
Learning Loss (inferred — confirm).
Revisit if a reviewer asks for full-pool results, or if query cost stops mattering.

# Architecture

How the pieces fit, and the seams that surprise someone who has only seen the file tree. Rules
that follow from these live in `CLAUDE.md`; why things are this way lives in `docs/decisions.md`.

## Method families

- **Offline GTG** (`gtg`, `ll_gtg` → `strategies/gtg/GTG_off.py`): train the backbone, then run GTG
  once over frozen embeddings of the labelled set plus the cycle's unlabelled pool, and query the
  highest aggregated entropy. `ll_gtg` differs only in training with the LossNet auxiliary.
- **Online GTG** (`llmlp_gtg`, `lsmlps_gtg`, `lsmlpsA_gtg`, `lstmreg_gtg`, `lstmbc_gtg` →
  `strategies/gtg/GTG.py`): `GTGModule` runs a differentiable GTG on each training batch, and a
  head from `GTG_Modules.py` (chosen by the method-name prefix) learns to predict the entropy
  signal, so acquisition trains jointly with the backbone. Querying ranks by the head.
  - The target (`y_true`) is GTG over the *detached* penultimate embeddings. Except for `llmlp`
    (a LossNet-shaped MLP), the prediction is *itself* a GTG run over learned projections
    (`Module_LS`) of the intermediate feature maps; that is how gradients reach `mod_ls`.
  - `lstmbc` is the odd one out: it predicts labelled-vs-unlabelled and queries the *lowest*
    scores (`largest=False`).
- Everything in `strategies/baselines/` and `strategies/competitors/` is a comparison method.
  Sequential GCN's `UncertainGCN`/`CoreGCN` share a `SequentialGCN` base.

## Run flow

`main.py` → per dataset: `Cls_Datasets` → one backbone (`dict_backbone` in `init.py`: ResNet18,
VGG16-bn for Caltech256) → `get_masters()` wraps it in one `Master_Model` per *module type* the
methods need → per trial: initial split → `get_strategies_object()` constructs **every** strategy
(GTG methods expanded over `-am` × `-am_ts`) → each strategy's `.run()` executes its full AL loop
in turn.

| Master | Module | Used by |
|---|---|---|
| `M_None` | none | baselines, competitors, `gtg` |
| `M_LL` | `LossNet`, 1 output | `ll`, `ll_gtg`, `tavaal` |
| `M_LL_tidal` | `LossNet`, `n_classes` outputs | `tidal` |
| `M_GTG_<id>` | `GTGModule` | one per online variant |

`ActiveLearner.run()` per cycle:
1. Draw a random unlabelled pool (`get_rand_unlab_sample`, size `unlab_sample_dim`).
2. `train_evaluate_save`: single GPU goes `Train_DDP.train` → `Worker.Cls_TrainWorker`, then
   writes `results.csv` and the train curves.
3. The subclass's `query(pool_subset, k) -> (pool-local indices, global indices)`.
4. `update_sets`, which hard-fails if the labelled / unlabelled / temp-pool index sets overlap.

Steps 3–4 are skipped in the last cycle: the loop ends there, so that query would select samples
no model is ever trained on (`count_classes.csv` therefore has `al_iters - 1` rows per trial).

A new strategy only implements `query()` (minimal shape: `strategies/baselines/Random.py`) and
registers in `get_args()` choices and a `dict_strategies_*` map in `init.py`.

`Master_Model.forward` switches on a mode argument and checks train/eval state. `'all'` only works
while training (and asserts embeddings are finite with std > 0); `'outs'`/`'embedds'`/`'module_out'`
only work in eval. `backbone.get_features()` returns the feature maps cached by the backbone's
*last* forward call.

**GTG core** (`GTG_Cls.py::GTGModule`, mirrored in `GTG_off.py`):
1. Build the affinity matrix from features (`get_A_cos_sim`/`get_A_corr`/`get_A_rbfk`, `-am`).
2. Sparsify (`-am_ts`, `-am_t`), then apply `-am_s` post-processing.
3. Scale the UL/LU blocks by the redundancy weight (`-am_r`).
4. Clamp labelled rows to their one-hot labels.
5. Iterate replicator dynamics until `-gtg_t` or `-gtg_iter`.
6. Aggregate each sample's entropy history with `-e_s` (`mean` or trapezoid `integral`).

Early convergence zero-pads the history, so slow-converging samples score higher. `-am_r` relies
on this.

**Config** (`app/config.py`) holds two separate things:
- `cls_datasets`: torchvision class, transforms, class names.
- `cls_config`/`al_params`: per-dataset batch size and optimizers keyed by module type
  (`None`/`'LossNet'`/`'GTGModule'`), plus the AL protocol constants.

`det_datasets`/`voc_config` are leftovers from the removed detection task. Cold start
(`-cs_s mse|kmeans`, backed by `app/auto_encoder/`) picks the initial labelled set and forces
`-tr 1`. `-bbone` requires it, and only loads the pre-trained encoder into online-GTG masters.

## Seams

### Side effects at import and construction time
- `main.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` *before* importing torch.
  Entry points that import `config` first (`smoke_run.py`, the tests) load torch before those are
  set.
- `init.py` imports every strategy module, so an import error in any competitor breaks every
  method.
- Constructing a `Master_Model` writes `app/checkpoints/<ds>/<model name>_init.pth.tar`. That path
  is cwd-relative, and the directories only exist because each holds a tracked `.gitignore`.
- Constructing a strategy creates its results directories. All strategies are built before the
  first one runs, so a crash in the first method still leaves folders for every method. Its initial
  class counts go to `_count_classes.csv` at the start of `run()`, not at construction, so that a
  strategy `main.py` skips or purges under `--existing` leaves no row behind.
- `Cls_Datasets(...)` downloads on first use, and `init_caltech256()` mutates
  `config.cls_datasets['caltech256']['transforms']` in place (list → `Compose`).
- `al_params`/`cls_config` are mutable module globals imported by name. `smoke_run.py` depends on
  this: it `.update()`s them before importing `main`. Rebinding them instead won't propagate.

### Shared mutable state across strategies
- Every master wraps the **same backbone object**, and all strategies of a module type share one
  master. This is only safe because each cycle's worker first loads the `_init` checkpoint and each
  `query()` first calls `load_best_checkpoint()`. A query that skips that call scores with
  whichever strategy trained last.
- **Online GTG affinity config: the last strategy constructed wins.** `GTG.__init__` calls
  `define_idx_params(id_am_ts, id_am)` on the shared `GTGModule`, and every strategy is constructed
  before any runs. With several `-am`/`-am_ts` values, all online strategies would train and query
  with the *last* combination while their names advertise their own, so `get_args()` refuses an
  online method whose `-am` × `-am_ts` combinations multiply out to more than one
  (`test_pipeline_fixes.py`). `GTG_off` keeps its config per instance and isn't affected.
- The Worker picks the module loss by matching the **`method_name` string** (`'LearningLoss'`,
  `'TA_VAAL'`, `'TiDAL'`, substring `'GTG_off'`, or a `GTGModule` master). That string is the
  strategy name with the `<model name>_` prefix stripped. Renaming a strategy class changes its
  loss or raises `Invalid method_name`. `ActiveLearner.run` uses the same `'GTG_off'` substring for
  its t-SNE call.

### Randomness
- `set_seeds()` resets everything to seed 10001 and runs at the start of every training cycle.
  Seeds used, each followed by a reset:
  - initial split: `dataset_id * trial`
  - each cycle's pool: `dataset_id * (trial * al_iters + cycle - 1)`

  Pools are identical across strategies in cycle 1 and diverge once their unlabelled lists differ.
- Plotting draws colours from a private `_PALETTE_RNG`. It used to shuffle the global `random`, so
  runs with and without `-diag` sampled different pools. Code on diagnostics-only paths must not
  touch the global RNGs.

### Datasets and batches
- The two dataset objects index the same train split with shared indices:

  | Object | Transforms | Yields | Used for |
  |---|---|---|---|
  | `train_ds` | train augmentations | `(idx, img, label, moving_prob)` | training |
  | `unlab_train_ds` | test transforms | `(idx, img, label)` | every query and embedding pass, labelled samples included |

  `moving_prob` is TiDAL's moving average, stored on the dataset and allocated with
  `share_memory_()`: the Worker updates it in the main process, and forked DataLoader workers (the
  Linux default, unlike macOS spawn) would otherwise keep reading a private, never-updated copy.
  `get_embeddings` tells the two apart by tuple length.
- There is no validation split. The "best" checkpoint (`check_best_path`, `load_best_checkpoint`)
  is simply the **last epoch**, and the reported number is that checkpoint's test accuracy, summed
  over correct predictions and divided by the test-set size (a mean of per-batch accuracies would
  over-weight the short last batch, differently for each batch size).
- Online-GTG batches put labelled samples first. The worker joins a labelled batch
  (`bsgtgo × cycle`) with an unlabelled one (`bsgtgo × al_iters`) and passes `n_lab_obs`; the module
  clamps rows `[:n_lab_obs]`. Training asserts both loaders yield the same number of batches
  (`zip` would silently truncate), while `GTG.query` only logs a warning.
- SVHN is deliberately unnormalised (code comment: normalisation "can produce black images").
- Caltech256 and Tiny-ImageNet have no smoke coverage, and from reading the code both look broken
  on first use:
  - `init_caltech256()` runs once per `Cls_Dataset` (three times), and the second call
    `.append`s to a `Compose`.
  - `download_tinyimagenet()` passes a single string to `subprocess.run` without `shell=True`.
  - Tiny-ImageNet's test split is read from the unlabelled `test/` folder.
  - Tiny-ImageNet class names come from unsorted `os.listdir`, while `ImageFolder` sorts them.

### Training schedule
- For every module-based method except TiDAL, from epoch `decay` (120) the module loss is
  multiplied by 0 and `Master_Model` detaches the features fed to the module. The module stops
  learning at epoch 120 of 200, so the `GTGModule` LR milestone (200) never takes effect.
- The `llmlp_gtg`/TiDAL exception in `Worker.train` can never trigger: its condition is always
  true.

### Two copies of the payoff-matrix code
`GTG_Cls.py` (per batch, differentiable) and `GTG_off.py` (whole pool, affinity built on CPU) each
implement `get_A_*`, thresholding, `diversity`/`mixed`, `-am_r` and the replicator loop. Several
fixes had to land in both: division by zero on isolated nodes, the rbfk distance clamp and
threshold, `1 - A` re-adding self-loops, and NaN from `corrcoef`. The copies already differ: only
`GTG_off` reads `-am_md`. `test_gtg_smoke.py` and `test_gtg_theory.py` exercise `GTGModule` only.

### Results, names and the paper pipeline
- Strategy name = `<backbone class>_<module name>_<method name>`. Examples:
  - `ResNet_Random`
  - `ResNet_LossNet_1_GTG_off_corr_mixed_ts-mean_es-mean`
  - `ResNet_GTGModule_lsmlps_GTG_mixed_corr_ts-mean_es-mean`
- Offline names put `<am>_<am_s>`; online names put `<am_s>_<am>`.
- Names encode `-am`, `-am_s`, `-am_ts` (plus `-am_t`, but only for `threshold`) and `-e_s`. They
  do **not** encode `-am_r`, `-am_md`, `-am_sl`, `-gtg_iter`, `-gtg_t` or `-bsgtgo`.
- The name keys results folders, `results.csv` rows and
  `app/checkpoints/<ds>/<name>_<device>.pth.tar`. Checkpoints aren't keyed by `-o`.
- Results layout, where `<exp>` is `-o` or a timestamp:
  - `results/<exp>/<ds>/<trial>/<name>/`: train/val curves, plus t-SNE, affinity and entropy plots
    only with `-diag`.
  - Dataset level: `results.csv`, `count_classes.csv`, `_count_classes.csv` (initial counts),
    `mean_std_clf_results.csv` and `yamal_conf.yamal`.
  - Experiment level, written by `run_sweep.py`: `logs/<ds>_<method>.log` and one
    `.sweep_<ds>_<method>.json` per method, holding its `status`, `trials` and `extra`.
  - The YAML is written only after every trial of a dataset finishes, so a sweep's file describes
    whichever process finished last.
- `results.csv` columns are `method, iter, lab_obs, test_accuracy`, but **`iter` holds the trial
  index** (same in the count CSVs). `write_csv` only appends; `main.py -ex/--existing` decides what
  happens to rows already there — `append` (default, duplicates them), `resume` (skip a complete
  strategy × trial, `purge_rows` a partial one and re-run it) or `overwrite`. `run_sweep.py` passes
  `resume`, or `overwrite` under `--force`.
- `docs/paper/make_results.py` maps each `method` to a display row by the first matching substring
  in `DISPLAY` (`gtg` and `ll_gtg` map to separate rows — the `LossNet_1_GTG_off` key has to
  precede `GTG_off`). `validate()` exits rather than build a table when rows are duplicated, when a
  strategy/budget cell doesn't have exactly `--trials` rows, or when one display row would pool
  several strategy names (several `-am` × `-am_ts` configurations, or several online variants):
  each of those inflates `count` and narrows the CI. The CI itself is Student-t, not z.
- `.gitignore` excludes `*.csv`, `*.png` and `*.log` everywhere; no results are versioned.
- Logging:
  - The logger writes only to `results/<exp>/exp_log.log`, in append mode, shared by every process
    of a sweep.
  - `run_sweep.py`'s per-method logs capture stdout/stderr, which is mostly tracebacks.
  - `GTGModule` still logs one summary line per batch, but its per-replicator-iteration state and
    its gradient hooks are DEBUG and off unless `main.py --debug_gtg` raises the two GTG loggers.
    Each of those formats a tensor and so synchronises with the GPU (~11 KB and ~136 syncs per
    training batch, ~3-4 GB of log over a CIFAR-10 sweep); they are the tool for tracing weights
    that diverge to NaN, and the NaN assertions stay on regardless. Also written at DEBUG: the same
    per-iteration state in `GTG_off`. The saved YAML records `git_commit` (short hash,
    `-dirty` when the checkout has uncommitted changes), which is also logged at run start.

### DDP path (work in progress)
`-gpus > 1` goes through `mp.spawn(train_ddp)`. Known gaps:
- it reads `SLURM_CPUS_PER_TASK`;
- it always builds a plain `DataLoader`, so online GTG's two-loader path is single-GPU only;
- it never sets `params["batch_size"]`, which the worker reads;
- it names checkpoints `cuda:<rank>`, while `load_best_checkpoint` looks for `cuda`.

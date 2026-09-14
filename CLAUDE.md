# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@docs/decisions.md
@docs/architecture.md

## What this is

ALGTG is a research codebase (MSc thesis + in-progress arXiv paper, see `docs/paper/`) for an Active
Learning query strategy based on the **Graph Transduction Game (GTG)**: pseudo-labels propagate
from a labelled to an unlabelled set via replicator dynamics, and each unlabelled sample's entropy
trajectory is the acquisition signal. Proposed methods: offline `gtg`, `ll_gtg`; online
`llmlp_gtg`, `lsmlps_gtg`, `lsmlpsA_gtg`, `lstmreg_gtg`, `lstmbc_gtg`. Comparison methods:
`random`, `entropy`, `ll`, `coreset`, `badge`, `bald`, `cdal`, `alphamix`, `tavaal`, `tidal`,
`uncertaingcn`, `coregcn`. How the code fits together is in `docs/architecture.md`. Operating a
sweep on a GPU host (launch, rollback, what to check when it dies) is in `docs/runbook.md`, which
is not imported: read it when you need it.

`get_args()` in `app/main.py` is the authoritative CLI reference. `README.md` explains the
affinity-matrix/entropy-strategy vocabulary, but its argument table and method list are stale
(missing `-o`, `-nw`, `-diag`, `-am_r` and the GCN methods). The AL protocol constants
(`init_lab_obs`, `al_iters`, `n_top_k_obs`, `unlab_sample_dim`) are `al_params` in `app/config.py`.

## Commands

Environment (conda, defined in `environment.yml`, env name `algtg`; edit the `prefix:` line to
your local conda path before creating it). The file was exported on macOS with exact build
strings and a pip `torch` wheel, so on a Linux CUDA host expect to relax the pins and install the
matching CUDA torch build. `get_device()` in `utils.py` picks CUDA → MPS → CPU.

```
conda env create -f environment.yml
conda activate algtg
```

**Run everything from the repo root.** Imports would resolve from any cwd (Python puts the
script's own directory, `app/`, on `sys.path` — hence prefix-less imports like
`from dataset import Cls_Datasets`), but checkpoints (`app/checkpoints/<dataset>/`), outputs
(`results/`) and downloads (`datasets/`) are cwd-relative paths. Never use `python -m app.xxx`;
always `python app/xxx.py` or `python app/tests/xxx.py`.

Run the benchmark (one invocation, one or more methods/datasets):
```
python app/main.py -gpus 1 -m random entropy coreset badge bald cdal tavaal ll ll_gtg -ds cifar10 -tr 5 -am corr -am_s mixed -am_ts mean -e_s mean
```
Only `-gpus 1` (single-GPU) is currently supported; the DDP path (`train_evaluate/Train_DDP.py`)
is a work in progress. `-diag` turns on expensive artefacts (t-SNE, affinity plots, per-sample
image dumps) — off by default because they can outweigh training cost on a full run. `-am` and
`-am_ts` accept several values and every GTG method is instantiated once per combination;
`-am_ts` **defaults to `mean none`**, so each GTG method runs twice unless you pass one value.

Resumable multi-method sweep (recommended for anything beyond a quick check — isolates each
method in its own process so one crash doesn't kill the run, and skips methods already marked
done on re-run):
```
python run_sweep.py -o run1 -ds cifar10 --trials 1 -m random     # time one method first
python run_sweep.py -o run1 -ds cifar10 --trials 5               # full sweep
python run_sweep.py -o run1 -ds cifar10 --status                 # progress
python run_sweep.py -o run1 -ds cifar10 -m gtg --extra -am_ts mean -e_s mean   # --extra must be last
```
Everything after `--extra` is forwarded verbatim to `main.py` (that's how GTG hyperparameters
reach a sweep). Per-method logs go to `results/<exp>/logs/<dataset>_<method>.log`, success writes
a `results/<exp>/.done_<dataset>_<method>` marker, and `--force` re-runs finished methods. The
default method list includes only `lsmlps_gtg` among the online GTG variants.

Tests live in `app/tests/` as plain scripts: each inserts `app/` onto `sys.path` itself and its
`__main__` block runs every `test_*` function. **`pytest` is not in `environment.yml`** — install
it to use `pytest app/tests/<file>.py::<test>`; without it, run a single test like the last line:
```
python app/tests/test_gtg_smoke.py         # core GTGModule end-to-end, CPU, seconds
python app/tests/test_gtg_theory.py        # guards the paper's theoretical claims (Prop. 1, redundancy term, etc.)
python app/tests/test_baseline_fixes.py    # regression tests for BALD/AlphaMix/BADGE/CDAL/TA-VAAL/GCN fixes
python app/tests/test_utils_fixes.py       # CSV/plotting/reporting-path regressions
python -c "import sys; sys.path.insert(0, 'app/tests'); import test_gtg_theory as t; t.test_proposition1_is_not_vacuous()"
```
Run the first four before any real sweep or before touching GTG math, baseline query logic, or
`utils.py` reporting code — they're what stands between a silent regression and a multi-day sweep
producing numbers that don't match what the paper claims.

`app/tests/smoke_run.py <method>` runs the real `main()` end-to-end on CIFAR-10 after shrinking
`al_params` and setting `epochs=1` in-process (downloads CIFAR-10 on first use, writes a
timestamped `results/` folder). It only proves the pipeline doesn't crash. All methods:
```
for m in random entropy coreset badge bald cdal alphamix uncertaingcn coregcn ll tidal tavaal gtg ll_gtg lsmlps_gtg; do python app/tests/smoke_run.py $m || echo "FAIL $m"; done
```

Turn a finished sweep's `results.csv` into the paper's table/figure (never hand-transcribe
numbers; missing method/budget cells are left as `--` rather than invented):
```
python docs/paper/make_results.py results/<exp_path> cifar10 --budgets 2000 4000 6000 8000 10000
```
Build the paper: `cd docs/paper && pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## Non-obvious constraints

- **Misspellings are part of the interface**: `uncertanity` (an `-am_s` value), `--gtg_tollerance`,
  `--affnity_matrix_self_loop`, `graph_trasduction_game`, `get_A_treshold`, `yamal_conf.yamal`.
  They flow into CLI choices, strategy names, `results.csv` and saved YAMLs — don't "fix" them in
  passing.
- **Proposition 1 (the paper states it, `test_gtg_theory.py` guards it)**: labelled players are
  clamped every iteration, so the LL and LU payoff blocks are inert. The default `-am_s mixed`
  (which inverts LL) therefore behaves like `uncertanity`, and `-am_md LL|LU` has no effect.
  `-am_md` is only read by `GTG_off`, where `UU`/`UL` do change the dynamics. `-am_r` (scales
  UL/LU) is the redundancy term that actually matters — don't design ablations around inert blocks.
- **Online-GTG loader alignment**: `Cls_TrainWorker` zips a labelled loader (batch
  `bsgtgo × AL cycle`) with an unlabelled one (batch `bsgtgo × al_iters`) and asserts they yield the
  same number of batches. Changing `al_params` or `-bsgtgo` can trip it; the defaults and
  `smoke_run.py` both satisfy `init_lab_obs == n_top_k_obs == unlab_sample_dim / al_iters`.
- **One configuration per `-o` when the numbers matter.** Strategy names don't encode `-am_r`,
  `-am_md`, `-am_sl`, `-gtg_iter`, `-gtg_t` or `-bsgtgo`, so runs differing only in those merge
  rows in `results.csv` (a `-am_r` sweep needs one `-o` per λ). `docs/paper/make_results.py` also pools
  every GTG config, and `gtg` with `ll_gtg`, into one table row. Re-running a method into an
  existing `-o` (including `--force`) appends duplicate rows rather than replacing them.
- **Online GTG methods get exactly one `-am` and one `-am_ts` value.** Their affinity config is
  set on a shared `GTGModule` when strategies are constructed, so with several values the last
  combination silently applies to all of them (the `-am_ts` default is two values).
- **Payoff-matrix changes go in both `GTG_Cls.py` and `GTG_off.py`.** The GTG tests only exercise
  `GTGModule`, so they won't catch an offline-only regression.
- **Never run two processes for the same dataset + method from one checkout at once**, including
  `smoke_run.py` next to a CIFAR-10 sweep. Checkpoints are cwd-relative and keyed by dataset +
  strategy name + device, not by `-o`.
- **Renaming a strategy class changes behaviour**: the Worker picks the training loss by matching
  the class-name string.

## Decision log

Before a session ends, if we settled a choice that had real alternatives, add an entry at the top
of `docs/decisions.md` in its format. That covers method or protocol design, a CLI or
results-schema change, a new dependency, a workaround chosen over a fix, and a fix that changes
reported numbers. Skip routine work: bug fixes with one obvious fix, refactors, tests, doc edits.
Mark rationale the user didn't state as "(inferred — confirm)".

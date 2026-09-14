# Runbook

There is no service to deploy. "Production" is a benchmark sweep on a GPU host, and its output is
`results/<exp>/`. Run commands from the repo root unless noted. Why things behave as described is
in `docs/architecture.md`.

## Deploy: launch a sweep

1. **Code on the host.** TODO: the real procedure (clone URL, which branch or commit, how the host
   is provisioned). Nothing records the commit a sweep ran from (the saved YAML has no hash).
   TODO: decide whether to record it.
2. **Environment.** `environment.yml` was exported on macOS (exact build strings, pip `torch`
   wheel). Edit `prefix:` first.
   ```
   conda env create -f environment.yml
   conda activate algtg
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```
   TODO: the exact commands that give a working CUDA torch on the Linux host (relaxed pins? which
   wheel index?).
3. **Datasets.** CIFAR-10/100, SVHN and Fashion-MNIST download into `datasets/` on first use. To
   fetch ahead of the run:
   ```
   python -c "import sys; sys.path.insert(0, 'app'); from dataset import Cls_Datasets; Cls_Datasets('cifar10')"
   ```
   TODO: a tested procedure for Tiny-ImageNet and Caltech256 (their loaders look broken; see
   architecture.md).
4. **Pre-flight.** Takes minutes and catches crashes before a multi-day run.
   ```
   python app/tests/test_gtg_smoke.py
   python app/tests/test_gtg_theory.py
   python app/tests/test_baseline_fixes.py
   python app/tests/test_utils_fixes.py
   for m in random entropy coreset badge bald cdal alphamix uncertaingcn coregcn ll tidal tavaal gtg ll_gtg lsmlps_gtg; do python app/tests/smoke_run.py $m || echo "FAIL $m"; done
   ```
   `smoke_run.py` writes to `app/checkpoints/cifar10/` under the real strategy names. Never run it
   while a CIFAR-10 sweep is running from the same checkout.
5. **Launch under tmux** (an SSH drop kills a bare process):
   ```
   tmux new -s sweep
   conda activate algtg
   python run_sweep.py -o <exp> -ds cifar10 --trials 1 -m random      # time one method
   python run_sweep.py -o <exp> -ds cifar10 --trials 5 --extra -am <am> -am_ts <ts> -e_s <es>
   ```
   Pass exactly one value per GTG option (see CLAUDE.md), and keep `--extra` last. Detach with
   `Ctrl-b d`; reattach with `tmux attach -t sweep`. TODO: record the flagship GTG configuration
   here once it's chosen.
6. **When it finishes:**
   ```
   python run_sweep.py -o <exp> -ds cifar10 --status
   python docs/paper/make_results.py results/<exp> cifar10 --budgets 2000 4000 6000 8000 10000
   ```
   Check `docs/paper/summary_cifar10.txt` against `docs/paper/RESULTS_PLAN.md` §1 before trusting any number.
   A `trials` value above `--trials` means pooled configs or duplicated rows.
   TODO: how results are copied off the host and archived.

## Rollback: undo bad results

`results.csv` is append-only and no tooling removes rows. The safe rollback is a **fresh `-o`**.

- **A bug invalidated a method's numbers.** Fix it, run the pre-flight tests, and re-run that
  method into a new `-o`. Don't `--force` into the old one: that appends new rows beside the
  invalid ones, and `make_results.py` averages both.
- **A method crashed mid-run.** Its finished cycles are already in `results.csv`, but it has no
  `.done_` marker. Re-running restarts it at trial 0 and duplicates those cycles. TODO: choose a
  procedure (re-run into a new `-o` and merge CSVs, or prune the method's rows first). Neither is
  scripted.
- **Re-run one finished method:** `rm results/<exp>/.done_<ds>_<method>`. The duplicate-row caveat
  applies here too.
- **Code:** TODO: nothing ties a results folder to a commit, so there's no defined way to find the
  code that produced it.

## Migrations: changes that invalidate or split existing results

There's no database. The "schema" is strategy names, CSV columns and checkpoint names.

- **Strategy name changes** (class names, the name format in `GTG.py`/`GTG_off.py`, backbone or
  module class names): new rows won't group with old ones, `make_results.py` may map them
  differently, and the Worker's loss dispatch may break. Use a new `-o`.
- **Anything names don't encode** (`-am_r`, `-am_md`, `-am_sl`, `-gtg_iter`, `-gtg_t`, `-bsgtgo`,
  `al_params`, `cls_config`, epochs): use a new `-o`, or rows merge silently.
- **`al_params` or `-bsgtgo` changes:** check the online-GTG loader invariant first.
  `smoke_run.py` overrides `al_params`, so it can't catch this. Set `b` to your `-bsgtgo`; the
  command should print `True`:
  ```
  python -c "import sys, math; sys.path.insert(0, 'app'); from config import al_params as p; b = 32; print(all(math.ceil((p['init_lab_obs'] + (c - 1) * p['n_top_k_obs']) / (b * c)) == math.ceil(p['unlab_sample_dim'] / (b * p['al_iters'])) for c in range(1, p['al_iters'] + 1)))"
  ```
- **Fixes to GTG math or baseline query logic** invalidate earlier results for the affected
  methods. Run the four test scripts, re-run into a new `-o`, and log it in `docs/decisions.md`.
- **Checkpoints never need migrating.** `_init` files are rewritten on every start, and within
  each cycle a strategy's checkpoint is rewritten before it's read.
- **`environment.yml` upgrades** (torch, numpy, pandas): TODO: an acceptance procedure beyond the
  test scripts, e.g. re-running one method for one trial against a reference folder.

## When a sweep is down

Start here:
```
tmux ls                                                  # is the session alive?
python run_sweep.py -o <exp> -ds <ds> --status           # which methods finished
tail -50 results/<exp>/logs/<ds>_<method>.log            # stdout/stderr: tracebacks
tail -100 results/<exp>/exp_log.log                      # detailed log, all methods, append mode
nvidia-smi                                               # GPU busy? memory?
df -h .                                                  # exp_log.log and datasets fill disks
```
Rows written so far, per method and trial. The `iter` column holds the trial; a complete trial has
`al_iters` = 10 rows.
```
python -c "import pandas as pd; df = pd.read_csv('results/<exp>/<ds>/results.csv'); print(df.groupby(['method', 'iter']).size())"
```

| Symptom | Likely cause | Action |
|---|---|---|
| `GTG online loaders are misaligned` | `al_params` / `-bsgtgo` break the invariant | see Migrations |
| `embedding is nan`, `std is zero or negative` | backbone training diverged | TODO: known remedies |
| `y_pred is nan`, `y_true is nan` | NaN in the GTG head or entropy | run `test_gtg_smoke.py`, `test_gtg_theory.py` |
| `NON EMPTY INDICES INTERSECTION`, or `list.remove(x): x not in list` in `update_sets` | a `query()` returned a duplicate or out-of-pool index | bug in that strategy's `query()` |
| `Invalid method_name` | a strategy class was renamed | architecture.md → shared mutable state |
| `FileNotFoundError` under `app/checkpoints/` | `main.py` not run from the repo root | `cd` to the root (`run_sweep.py` sets cwd itself) |
| `KeyError: 'SLURM_CPUS_PER_TASK'` | `-gpus` > 1 | only `-gpus 1` is supported |
| CUDA out of memory | `cls_config` batch sizes; the online batch is `bsgtgo × (cycle + al_iters)` and grows each cycle | TODO: tested lower settings per dataset (changing `-bsgtgo` interacts with the invariant) |
| `OMP: Error #15` (macOS) | duplicate OpenMP runtime | `export KMP_DUPLICATE_LIB_OK=TRUE` (`main.py` and `environment.yml` set it) |
| Process gone, no traceback | killed from outside (OOM killer, preemption, no tmux) | `dmesg -T \| grep -i -E 'killed process\|out of memory'`; TODO: provider-side preemption check |
| No progress, `exp_log.log` still growing | slow phase: `GTG_off` query on 11k–20k nodes, or `-diag` t-SNE | wait, or drop `-diag` |
| No progress, log silent | DataLoader workers hung | re-run with `--num_workers 0` |

After fixing, re-run the same `run_sweep.py` command. Finished methods are skipped; see Rollback
for the partial rows of the method that crashed.

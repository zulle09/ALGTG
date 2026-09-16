#!/usr/bin/env python3
"""
Resumable benchmark sweep: runs one method per process into a shared results folder.

Why not just call main.py with every method at once: a crash (OOM, preemption, a bug in one
strategy) kills the whole invocation and the remaining methods never run. Here each method is
its own process, a failure is recorded and the sweep carries on, and re-running the same
command skips whatever already finished. On a rented GPU that difference is money.

    # estimate first: how long does one method take? (its own -o, so it can't mix with the sweep)
    python run_sweep.py -o timing -ds cifar10 -m random --trials 1

    # then the real sweep (safe to re-run after a crash or a disconnect)
    python run_sweep.py -o run1 -ds cifar10 --trials 5

    # inspect progress at any time
    python run_sweep.py -o run1 -ds cifar10 --status

A method that crashed or was interrupted is re-launched with `--existing resume`: main.py skips
the trials already complete in results.csv, discards the partial one and re-runs it. Each method's
trials and --extra are recorded in results/<exp>/.sweep_<dataset>_<method>.json, and a re-run
with a different --extra or fewer --trials is refused, because its rows would mix with the old ones.

Always launch it under tmux/nohup so an SSH drop does not kill the run.
"""

import argparse
import json
import os
import subprocess
import sys
import time

ALL_METHODS = [
    "random", "entropy", "coreset", "badge", "bald", "cdal", "alphamix",
    "uncertaingcn", "coregcn", "ll", "tavaal", "tidal",
    "gtg", "ll_gtg", "lsmlps_gtg",
]

HERE = os.path.dirname(os.path.abspath(__file__))


def state_path(exp_path: str, dataset: str, method: str) -> str:
    return os.path.join(HERE, "results", exp_path, f".sweep_{dataset}_{method}.json")


def legacy_marker(exp_path: str, dataset: str, method: str) -> str:
    return os.path.join(HERE, "results", exp_path, f".done_{dataset}_{method}")


def read_state(path: str):
    if not os.path.exists(path): return None
    with open(path) as f: return json.load(f)


def write_state(path: str, status: str, trials: int, extra) -> None:
    with open(path, "w") as f: json.dump({"status": status, "trials": trials, "extra": extra}, f)


def fmt(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h{rem // 60:02d}m" if h else f"{rem // 60}m{rem % 60:02d}s"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--exp_path", required=True, help="shared output folder under results/")
    ap.add_argument("-ds", "--dataset", required=True)
    ap.add_argument("-m", "--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="everything after --extra is forwarded verbatim to main.py")
    ap.add_argument("--status", action="store_true", help="print progress and exit")
    ap.add_argument("--force", action="store_true",
                    help="re-run methods already done with the same --trials and --extra, replacing their rows")
    a = ap.parse_args()

    out = os.path.join(HERE, "results", a.exp_path)
    os.makedirs(out, exist_ok=True)
    log_dir = os.path.join(out, "logs")
    os.makedirs(log_dir, exist_ok=True)

    if a.status:
        for m in a.methods:
            state = read_state(state_path(a.exp_path, a.dataset, m))
            if state is None: label = "--"
            elif state["status"] == "done": label = f"done ({state['trials']} trials)"
            else: label = "incomplete"
            print(f"  {label:>16}  {m}")
        return

    # Validate every method before launching any, so a mismatch is reported before paid time is spent.
    todo, refused = [], []
    for m in a.methods:
        if os.path.exists(legacy_marker(a.exp_path, a.dataset, m)):
            refused.append(f"{m}: has a .done_ marker from an older run_sweep that recorded no configuration")
            continue
        state = read_state(state_path(a.exp_path, a.dataset, m))
        if state is not None:
            if state["extra"] != a.extra:
                refused.append(f"{m}: recorded with --extra {' '.join(state['extra']) or '(none)'}, "
                               f"requested --extra {' '.join(a.extra) or '(none)'}")
                continue
            if a.trials < state["trials"]:
                refused.append(f"{m}: recorded with --trials {state['trials']}, requested {a.trials}")
                continue
            if state["status"] == "done" and state["trials"] == a.trials and not a.force:
                continue
        todo.append(m)

    if refused:
        print(f"refusing to run: rows of these methods already in '{a.exp_path}' would mix with a "
              "different setup. Use a fresh -o, or repeat the recorded --trials/--extra:")
        for reason in refused: print(f"  {reason}")
        sys.exit(2)

    skipped = len(a.methods) - len(todo)
    print(f"sweep '{a.exp_path}' on {a.dataset}: {len(todo)} to run, {skipped} already done\n")

    failed, durations = [], []
    for i, method in enumerate(todo, 1):
        log_path = os.path.join(log_dir, f"{a.dataset}_{method}.log")
        cmd = [sys.executable, os.path.join(HERE, "app", "main.py"),
               "-m", method, "-ds", a.dataset, "-tr", str(a.trials),
               "-o", a.exp_path, "-nw", str(a.num_workers), "-gpus", "1",
               "--existing", "overwrite" if a.force else "resume"] + a.extra

        eta = f" | eta ~{fmt(sum(durations) / len(durations) * (len(todo) - i + 1))}" if durations else ""
        print(f"[{i}/{len(todo)}] {method}{eta}", flush=True)

        path = state_path(a.exp_path, a.dataset, method)
        write_state(path, "running", a.trials, a.extra)
        t0 = time.time()
        with open(log_path, "a") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
        dt = time.time() - t0

        if rc == 0:
            write_state(path, "done", a.trials, a.extra)
            durations.append(dt)
            print(f"        ok in {fmt(dt)}", flush=True)
        else:
            failed.append(method)
            print(f"        FAILED (exit {rc}) after {fmt(dt)} -> {log_path}", flush=True)

    print(f"\ndone: {len(todo) - len(failed)}/{len(todo)} succeeded")
    if failed:
        print(f"failed: {', '.join(failed)}")
        print("inspect the logs, fix, then re-run the same command — finished trials are skipped")
    print(f"\nnext:  python docs/paper/make_results.py results/{a.exp_path} {a.dataset} --trials {a.trials}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

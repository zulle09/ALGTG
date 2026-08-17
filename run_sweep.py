#!/usr/bin/env python3
"""
Resumable benchmark sweep: runs one method per process into a shared results folder.

Why not just call main.py with every method at once: a crash (OOM, preemption, a bug in one
strategy) kills the whole invocation and the remaining methods never run. Here each method is
its own process, a failure is recorded and the sweep carries on, and re-running the same
command skips whatever already finished. On a rented GPU that difference is money.

    # estimate first: how long does one method take?
    python run_sweep.py -o run1 -ds cifar10 -m random --trials 1

    # then the real sweep (safe to re-run after a crash or a disconnect)
    python run_sweep.py -o run1 -ds cifar10 --trials 5

    # inspect progress at any time
    python run_sweep.py -o run1 -ds cifar10 --status

Always launch it under tmux/nohup so an SSH drop does not kill the run.
"""

import argparse
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


def marker(exp_path: str, dataset: str, method: str) -> str:
    return os.path.join(HERE, "results", exp_path, f".done_{dataset}_{method}")


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
    ap.add_argument("--force", action="store_true", help="re-run methods already marked done")
    a = ap.parse_args()

    out = os.path.join(HERE, "results", a.exp_path)
    os.makedirs(out, exist_ok=True)
    log_dir = os.path.join(out, "logs")
    os.makedirs(log_dir, exist_ok=True)

    if a.status:
        for m in a.methods:
            print(f"  {'done' if os.path.exists(marker(a.exp_path, a.dataset, m)) else '  --':>5}  {m}")
        return

    todo = [m for m in a.methods
            if a.force or not os.path.exists(marker(a.exp_path, a.dataset, m))]
    skipped = len(a.methods) - len(todo)
    print(f"sweep '{a.exp_path}' on {a.dataset}: {len(todo)} to run, {skipped} already done\n")

    failed, durations = [], []
    for i, method in enumerate(todo, 1):
        log_path = os.path.join(log_dir, f"{a.dataset}_{method}.log")
        cmd = [sys.executable, os.path.join(HERE, "app", "main.py"),
               "-m", method, "-ds", a.dataset, "-tr", str(a.trials),
               "-o", a.exp_path, "-nw", str(a.num_workers), "-gpus", "1"] + a.extra

        eta = f" | eta ~{fmt(sum(durations) / len(durations) * (len(todo) - i + 1))}" if durations else ""
        print(f"[{i}/{len(todo)}] {method}{eta}", flush=True)

        t0 = time.time()
        with open(log_path, "w") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
        dt = time.time() - t0

        if rc == 0:
            open(marker(a.exp_path, a.dataset, method), "w").close()
            durations.append(dt)
            print(f"        ok in {fmt(dt)}", flush=True)
        else:
            failed.append(method)
            print(f"        FAILED (exit {rc}) after {fmt(dt)} -> {log_path}", flush=True)

    print(f"\ndone: {len(todo) - len(failed)}/{len(todo)} succeeded")
    if failed:
        print(f"failed: {', '.join(failed)}")
        print("inspect the logs, fix, then re-run the same command — finished methods are skipped")
    print(f"\nnext:  python paper/make_results.py results/{a.exp_path} {a.dataset}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

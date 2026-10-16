"""
End-to-end smoke run: exercises the REAL main() on a miniature AL protocol.

Its only job is to prove the whole pipeline (dataset -> training -> query -> set update ->
plots/CSV) runs to completion for a given method, so that a multi-day sweep does not die on
a trivial crash. It is not a correctness or accuracy check.

The AL protocol is shrunk so a cycle takes seconds. The sizes keep the GTG-online loader
alignment valid (init_lab_obs == unlab_sample_dim / al_iters, and n_top_k_obs == init_lab_obs),
which is what the assertion in Cls_TrainWorker checks.

    python app/tests/smoke_run.py <method> [main.py flags]   # one method, isolated; e.g. -fs file -fp feats.npy
    for m in random entropy coreset ...; do python app/tests/smoke_run.py $m || echo FAIL $m; done
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import al_params, cls_config


DATASET = "cifar10"
SMOKE_AL = dict(init_lab_obs=64, al_iters=2, n_top_k_obs=64, unlab_sample_dim=128)
SMOKE_BSGTGO = 8          # lab batch = 8*iter, unlab batch = 8*al_iters -> 8 batches either way


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    method, extra_args = sys.argv[1], sys.argv[2:]

    al_params.update(SMOKE_AL)
    cls_config["epochs"] = 1
    cls_config[DATASET]["batch_size"] = {None: 32, "LossNet": 32, "GTGModule": 32}

    sys.argv = [
        "main.py", "-gpus", "1", "-m", method, "-ds", DATASET, "-tr", "1",
        "-bsgtgo", str(SMOKE_BSGTGO), "-am", "corr", "-am_ts", "mean",
    ] + extra_args

    import main as app_main
    app_main.main()
    print(f"SMOKE OK: {method}")


if __name__ == "__main__":
    main()

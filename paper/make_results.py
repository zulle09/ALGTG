"""
Turn the benchmark output into the paper's table and figure.

Reads `results/<exp_path>/<dataset>/results.csv` (written by ActiveLearner.train_evaluate_save)
and emits, into this folder:

    results_<dataset>.tex   - booktabs table: mean +/- 95% CI per method per budget, best in bold
    curve_<dataset>.pdf     - accuracy-vs-budget curves with CI bands
    summary_<dataset>.txt   - AUBC ranking, for the prose

Usage:
    python paper/make_results.py results/2026-07-14_10-00-00 cifar10
    python paper/make_results.py results/<exp> cifar10 --budgets 2000 5000 10000

Nothing here fabricates numbers: if the CSV is missing a method/budget the cell is left as '--'.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Presentation order and display names. Methods absent from the CSV are skipped.
DISPLAY = [
    ("Random",        "Random"),
    ("Entropy",       "Entropy"),
    ("CoreSet",       "Core-Set"),
    ("BADGE",         "BADGE"),
    ("BALD",          "BALD"),
    ("CDAL",          "CDAL"),
    ("AlphaMix",      "AlphaMix"),
    ("LearningLoss",  "Learning Loss"),
    ("TA_VAAL",       "TA-VAAL"),
    ("TiDAL",         "TiDAL"),
    ("UncertainGCN",  "UncertainGCN"),
    ("CoreGCN",       "CoreGCN"),
    ("GTG_off",       "ALGTG (offline)"),
    ("GTGModule",     "ALGTG (ours)"),
]


def _short_name(strategy_name: str) -> str:
    """'ResNet_GTGModule_lsmlps_mixed_corr...' -> the discriminating middle token."""
    for key, _ in DISPLAY:
        if key in strategy_name:
            return key
    return strategy_name


def load(exp_path: str, dataset: str) -> pd.DataFrame:
    csv = os.path.join(exp_path, dataset, "results.csv")
    if not os.path.exists(csv):
        sys.exit(f"No results at {csv}. Run the benchmark first — this script never invents numbers.")
    df = pd.read_csv(csv)
    df["short"] = df["method"].map(_short_name)
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # accuracy_score() returns a fraction; the paper reports percentages
    if df["test_accuracy"].max() <= 1.0: df["test_accuracy"] = df["test_accuracy"] * 100.
    g = df.groupby(["short", "lab_obs"])["test_accuracy"].agg(["mean", "std", "count"]).reset_index()
    g["ci"] = 1.96 * g["std"].fillna(0.) / np.sqrt(g["count"].clip(lower=1))
    return g


def write_table(g: pd.DataFrame, budgets, dataset: str, out_dir: str) -> str:
    present = [(k, lbl) for k, lbl in DISPLAY if k in set(g["short"])]
    budgets = [b for b in budgets if b in set(g["lab_obs"])]

    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{Test accuracy (\%) on {dataset.upper()} at increasing labelling budgets. "
        r"Mean $\pm$ 95\% CI over the trials; best per column in bold.}",
        rf"\label{{tab:main-{dataset}}}",
        r"\begin{tabular}{l" + "c" * len(budgets) + "}", r"\toprule",
        "Method & " + " & ".join(str(b) for b in budgets) + r" \\", r"\midrule",
    ]

    best = {}
    for b in budgets:
        sub = g[g["lab_obs"] == b]
        if not sub.empty: best[b] = sub.loc[sub["mean"].idxmax(), "short"]

    for key, label in present:
        cells = []
        for b in budgets:
            row = g[(g["short"] == key) & (g["lab_obs"] == b)]
            if row.empty:
                cells.append("--"); continue
            m, ci = float(row["mean"].iloc[0]), float(row["ci"].iloc[0])
            cell = f"{m:.2f}\\pm{ci:.2f}"
            cells.append(rf"$\mathbf{{{cell}}}$" if best.get(b) == key else f"${cell}$")
        if "ALGTG" in label: lines.append(r"\midrule")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path = os.path.join(out_dir, f"results_{dataset}.tex")
    with open(path, "w") as f: f.write("\n".join(lines))
    return path


def write_curve(g: pd.DataFrame, dataset: str, out_dir: str) -> str:
    present = [(k, lbl) for k, lbl in DISPLAY if k in set(g["short"])]
    plt.figure(figsize=(7, 5))
    for key, label in present:
        sub = g[g["short"] == key].sort_values("lab_obs")
        if sub.empty: continue
        ours = "ALGTG" in label
        plt.plot(sub["lab_obs"], sub["mean"], label=label,
                 lw=2.4 if ours else 1.3, zorder=3 if ours else 2,
                 marker="o" if ours else None, ms=4,
                 color="crimson" if label == "ALGTG (ours)" else None)
        plt.fill_between(sub["lab_obs"], sub["mean"] - sub["ci"], sub["mean"] + sub["ci"], alpha=0.15)

    plt.xlabel("# labelled observations"); plt.ylabel("Test accuracy (%)")
    plt.title(f"{dataset.upper()}"); plt.grid(alpha=0.3); plt.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, f"curve_{dataset}.pdf")
    plt.savefig(path); plt.close()
    return path


def write_summary(g: pd.DataFrame, dataset: str, out_dir: str) -> str:
    """Area under the budget-accuracy curve — the single number to rank methods by."""
    rows = []
    for key in g["short"].unique():
        sub = g[g["short"] == key].sort_values("lab_obs")
        if len(sub) < 2: continue
        trapz = getattr(np, "trapezoid", None) or np.trapz    # renamed in numpy 2.0
        aubc = trapz(sub["mean"].values, sub["lab_obs"].values) / (
            sub["lab_obs"].max() - sub["lab_obs"].min())
        rows.append((key, aubc, sub["mean"].iloc[-1], int(sub["count"].min())))
    rows.sort(key=lambda r: -r[1])

    out = [f"{dataset.upper()} — AUBC ranking (higher is better)", "-" * 52,
           f"{'method':<20}{'AUBC':>10}{'final':>10}{'trials':>8}"]
    out += [f"{k:<20}{a:>10.3f}{f:>10.2f}{n:>8}" for k, a, f, n in rows]
    path = os.path.join(out_dir, f"summary_{dataset}.txt")
    with open(path, "w") as fh: fh.write("\n".join(out) + "\n")
    print("\n".join(out))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_path", help="e.g. results/2026-07-14_10-00-00")
    ap.add_argument("dataset")
    ap.add_argument("--budgets", type=int, nargs="+", default=[2000, 4000, 6000, 8000, 10000])
    a = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    df = load(a.exp_path, a.dataset)
    g = aggregate(df)

    print(f"loaded {len(df)} rows | methods: {sorted(g['short'].unique())}")
    print("wrote", write_table(g, a.budgets, a.dataset, out_dir))
    print("wrote", write_curve(g, a.dataset, out_dir))
    print("wrote", write_summary(g, a.dataset, out_dir))

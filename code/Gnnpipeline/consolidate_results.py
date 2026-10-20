#!/usr/bin/env python3
"""Consolidate per-dataset benchmark CSVs into one final table.

``run_sweep.py`` appends one row per (run) to ``results/<dataset>_benchmark.csv``,
each carrying ``Notes="seed=<s> epochs=<n>"``.  This script aggregates across seeds
per (Dataset, Model) and reports:

  * **mean +/- std** of ROC-AUC / Precision / Recall / F1 across seeds (the honest
    estimate of model quality);
  * **best** and **worst** per seed, for reference.

It writes ``model_benchmarks_results.csv`` (mean-based) plus a plain-text summary.

Usage
-----
    python consolidate_results.py [--logdir results] [--output model_benchmarks_results.csv]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline"))
from pipeline.metrics import Result  # noqa: E402

MEAN_COLS = ["Dataset", "Model", "Precision", "Recall", "False Positives",
             "False Negatives", "F1-Score", "ROC-AUC", "N_Support"]
STD_COLS = ["Precision Std", "Recall Std", "F1 Std", "ROC-AUC Std"]


def _mean_std(vals):
    vals = [float(v) for v in vals]
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / max(1, n - 1)
    return mean, (var ** 0.5 if n > 1 else 0.0)


def _stds(rows):
    """Return ``{col: sample-std}`` for the per-seed columns in ``STD_COLS``."""
    return {c: _mean_std([float(r[c]) for r in rows if r.get(c)])[1] for c in STD_COLS}


def _parse_seed(row):
    notes = row.get("Notes", "")
    for tok in notes.split():
        if tok.startswith("seed="):
            try:
                return int(tok.split("=", 1)[1])
            except ValueError:
                return None
    return None


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default="results")
    p.add_argument("--output", default="model_benchmarks_results.csv")
    a = p.parse_args()

    csvs = sorted(glob.glob(os.path.join(a.logdir, "*_benchmark.csv")))
    if not csvs:
        print("No per-dataset CSVs found in", a.logdir, file=sys.stderr)
        sys.exit(1)

    # group rows by (dataset, model)
    groups = {}
    for path in csvs:
        ds = os.path.basename(path)[:-len("_benchmark.csv")]
        for r in load_csv(path):
            key = (r["Dataset"], r["Model"])
            groups.setdefault(key, []).append(r)

    consolidated = []
    for key, rows in groups.items():
        seeds = sorted({_parse_seed(r) for r in rows if _parse_seed(r) is not None})
        n_runs = len(rows)
        # mean / std across seeds
        mean, std = _mean_std([float(r["ROC-AUC"]) for r in rows])
        std_cols = _stds(rows)
        consolidated.append({
            "Dataset": key[0], "Model": key[1],
            "Precision": f"{_mean_std([float(r['Precision']) for r in rows])[0]:.4f}",
            "Recall": f"{_mean_std([float(r['Recall']) for r in rows])[0]:.4f}",
            "False Positives": int(_mean_std([float(r['False Positives']) for r in rows])[0]),
            "False Negatives": int(_mean_std([float(r['False Negatives']) for r in rows])[0]),
            "F1-Score": f"{_mean_std([float(r['F1-Score']) for r in rows])[0]:.4f}",
            "ROC-AUC": f"{mean:.4f}", "N_Support": rows[0].get("N_Support", ""),
            "N_Runs": n_runs, "Seeds": ",".join(map(str, seeds)),
            **std_cols,
        })
        # best / worst per seed (reference)
        best = max(rows, key=lambda r: float(r["ROC-AUC"]))
        worst = min(rows, key=lambda r: float(r["ROC-AUC"]))
        print(f"{key[0]:<12} {key[1]:<16} AUC {mean:.3f} +/- {std:.3f} "
              f"(n={n_runs}, seeds={seeds}) "
              f"best {float(best['ROC-AUC']):.3f} / worst {float(worst['ROC-AUC']):.3f}")

    consolidated.sort(key=lambda r: (r["Dataset"], r["Model"]))

    with open(a.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MEAN_COLS + STD_COLS)
        w.writeheader()
        for r in consolidated:
            w.writerow({c: r.get(c, "") for c in MEAN_COLS + STD_COLS})
    print(f"\nWrote {len(consolidated)} rows -> {a.output}")

    print("\n=== FINAL BENCHMARK (mean +/- std across seeds) ===")
    print(f"{'Dataset':<12}{'Model':<16}{'Prec':<8}{'Recall':<8}"
          f"{'FP':<5}{'FN':<5}{'F1':<8}{'ROC-AUC':<10}{'#runs':<7}{'AUC Std':<9}")
    print("-" * len("=".ljust(81)))
    for r in consolidated:
        print(f"{r['Dataset']:<12}{r['Model']:<16}{float(r['Precision']):<8.3f}"
              f"{float(r['Recall']):<8.3f}{int(r['False Positives']):<5}"
              f"{int(r['False Negatives']):<5}{float(r['F1-Score']):<8.3f}"
              f"{float(r['ROC-AUC']):<10.3f}{int(r.get('N_Runs', 0)):<7}{float(r.get('ROC-AUC Std', 0)):<9.3f}")


if __name__ == "__main__":
    main()

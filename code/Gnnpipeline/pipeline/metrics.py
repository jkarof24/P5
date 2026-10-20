"""Standardized classification metrics and CSV result logging.

All models write into a single table: ``model_benchmarks_results.csv`` with
columns:
    Dataset, Model, Precision, Recall, False Positives, False Negatives,
    F1-Score, ROC-AUC
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, asdict, field

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

METRIC_COLS = [
    "precision",
    "recall",
    "false_positives",
    "false_negatives",
    "f1",
    "roc_auc",
]

# Canonical column order for every ``*_benchmark.csv`` (per-dataset and the
# consolidated table). All writers (``write_csv``, ``ensure_ds`` in run_sweep.py
# and ``consolidate_results.py``) must use exactly this order so the rows line
# up under ``csv.DictReader``.
CSV_FIELDNAMES = [
    "Dataset", "Model", "Precision", "Recall", "False Positives", "False Negatives",
    "F1-Score", "ROC-AUC", "N_Support", "Notes",
]


@dataclass
class Result:
    dataset: str
    model: str
    precision: float
    recall: float
    false_positives: int
    false_negatives: int
    f1: float
    roc_auc: float
    n_support: int = 0
    notes: str = ""

    def to_row(self):
        d = {
            "Dataset": self.dataset,
            "Model": self.model,
            "Precision": f"{self.precision:.4f}",
            "Recall": f"{self.recall:.4f}",
            "False Positives": int(self.false_positives),
            "False Negatives": int(self.false_negatives),
            "F1-Score": f"{self.f1:.4f}",
            "ROC-AUC": f"{self.roc_auc:.4f}",
        }
        if self.n_support:
            d["N_Support"] = int(self.n_support)
        if self.notes:
            d["Notes"] = self.notes
        return d


def compute(
    y_true,
    y_score,
    pos_label=1,
):
    """Compute all benchmark metrics from integer labels and class-probabilities.

    Returns a :class:`Result`. Raises if the score column has <2 positives or
    <2 negatives (ROC-AUC / AUC-PR are undefined).
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    y_true = y_true.astype(int)
    y_score = y_score[:, 1] if y_score.ndim > 1 else y_score

    # AUC is only defined when there are at least two of each class to rank;
    # sklearn raises otherwise. When it cannot be computed we report 0.5 (coin
    # flip) instead of nan so downstream tables stay numerically clean. NOTE:
    # this is a class-COUNT test, NOT a 0.5-threshold test -- a confident model
    # whose scores all sit on one side of 0.5 still has a well-defined AUC and
    # must not be forced to 0.5 (which the previous implementation did).
    n_pos = int((y_true == pos_label).sum())
    n_neg = int((y_true != pos_label).sum())
    if n_pos >= 2 and n_neg >= 2:
        auc = float(roc_auc_score(y_true, y_score))
    else:
        auc = 0.5

    # Precision / recall / F1 / FP / FN
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, (y_score > 0.5).astype(int),
        pos_label=pos_label, zero_division=0, average="binary",
    )
    fp = int(confusion_matrix(y_true, (y_score > 0.5).astype(int), labels=[0, 1])[0, 1])
    fn = int(confusion_matrix(y_true, (y_score > 0.5).astype(int), labels=[0, 1])[1, 0])

    return Result(
        dataset="", model="",
        precision=float(precision), recall=float(recall),
        false_positives=fp, false_negatives=fn,
        f1=float(f1), roc_auc=auc,
        n_support=int(y_true.size),
    )


def write_csv(results, path=None):
    """Append (or initialise) the standardized benchmark CSV using the canonical
    column order (``CSV_FIELDNAMES``). Writes the header only when the file is
    new/empty, and appends each result row in that fixed order -- so multiple
    appenders (the per-dataset sweep loop) never emit a duplicate/stale header
    that would corrupt downstream ``DictReader`` parsing.
    """
    if path is None:
        from pipeline.config import result_csv_path
        path = result_csv_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not (os.path.exists(path) and os.path.getsize(path) > 0)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_FIELDNAMES))
        if new:
            w.writeheader()
        for r in results:
            w.writerow(r.to_row())
    return path


def summary_table(results):
    """Return a compact str table for console reporting."""
    header = (
        f"{'Dataset':<12}{'Model':<14}"
        f"{'Prec':<7}{'Recall':<8}{'FP':<5}{'FN':<5}"
        f"{'F1':<7}{'ROC-AUC':<9}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.dataset:<12}{r.model:<14}"
            f"{r.precision:<7.3f}{r.recall:<8.3f}{r.false_positives:<5}"
            f"{r.false_negatives:<5}{r.f1:<7.3f}{r.roc_auc:<9.3f}"
        )
    return "\n".join(lines)

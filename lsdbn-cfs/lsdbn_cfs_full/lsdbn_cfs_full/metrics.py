from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def mean_std(values) -> str:
    arr = np.asarray(values, dtype=np.float64)
    return "%.4f +/- %.4f" % (float(arr.mean()), float(arr.std(ddof=0)))


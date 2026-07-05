"""Model evaluation and classification into Excel/Average/Struggle buckets."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return standard regression metrics."""

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classify_performance(
    scores,
    *,
    excel_threshold: float = 75.0,
    struggle_threshold: float = 40.0,
) -> np.ndarray:
    """Convert scores into Excel, Average, or Struggle labels."""

    scores = np.asarray(scores, dtype=float)
    labels = np.full(scores.shape, "Average", dtype=object)
    labels[scores >= excel_threshold] = "Excel"
    labels[scores <= struggle_threshold] = "Struggle"
    return labels


def classification_metrics(
    y_true,
    y_pred,
    *,
    labels: tuple[str, ...] = ("Excel", "Average", "Struggle"),
) -> dict[str, object]:
    """Return balanced and per-class classification metrics."""

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(labels),
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            label: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for label, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)),
    }


def optimize_thresholds(
    y_true_scores: np.ndarray,
    y_pred_scores: np.ndarray,
    *,
    struggle_range: tuple[float, float] = (30.0, 50.0),
    excel_range: tuple[float, float] = (65.0, 85.0),
    step: float = 1.0,
    metric: str = "macro_f1",
) -> dict[str, object]:
    """Grid search over classification thresholds to maximize a given metric.

    Finds the optimal struggle_threshold and excel_threshold that maximize
    the chosen metric (e.g., macro F1, balanced accuracy) on validation data.

    Parameters
    ----------
    y_true_scores : np.ndarray
        True continuous performance scores.
    y_pred_scores : np.ndarray
        Predicted continuous performance scores.
    struggle_range : tuple[float, float]
        Range to search for the struggle threshold (default 30-50).
    excel_range : tuple[float, float]
        Range to search for the excel threshold (default 65-85).
    step : float
        Step size for grid search (default 1.0).
    metric : str
        Metric to optimize: 'macro_f1', 'balanced_accuracy', or 'struggle_recall'.

    Returns
    -------
    dict with keys:
        - 'struggle_threshold': optimal struggle threshold
        - 'excel_threshold': optimal excel threshold
        - 'best_score': best metric value achieved
        - 'all_results': list of (struggle_th, excel_th, score) for analysis
    """
    true_labels = classify_performance(y_true_scores)
    best_score = -np.inf
    best_struggle = struggle_range[0]
    best_excel = excel_range[0]
    all_results = []

    struggle_candidates = np.arange(struggle_range[0], struggle_range[1] + step, step)
    excel_candidates = np.arange(excel_range[0], excel_range[1] + step, step)

    for st in struggle_candidates:
        for et in excel_candidates:
            if st >= et:
                continue  # Struggle threshold must be below Excel threshold

            pred_labels = classify_performance(y_pred_scores, struggle_threshold=st, excel_threshold=et)

            if metric == "macro_f1":
                score = float(f1_score(true_labels, pred_labels, average="macro", zero_division=0))
            elif metric == "balanced_accuracy":
                score = float(balanced_accuracy_score(true_labels, pred_labels))
            elif metric == "struggle_recall":
                # Focus specifically on Struggle class recall
                _, recall, _, _ = precision_recall_fscore_support(
                    true_labels, pred_labels, labels=["Struggle"], zero_division=0
                )
                score = float(recall[0]) if len(recall) > 0 else 0.0
            else:
                raise ValueError(f"Unknown metric: {metric}")

            all_results.append((float(st), float(et), score))

            if score > best_score:
                best_score = score
                best_struggle = st
                best_excel = et

    return {
        "struggle_threshold": float(best_struggle),
        "excel_threshold": float(best_excel),
        "best_score": float(best_score),
        "metric": metric,
        "all_results": all_results,
    }
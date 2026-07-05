"""Model evaluation and classification into Excel/Average/Struggle buckets."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
	accuracy_score,
	balanced_accuracy_score,
	confusion_matrix,
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
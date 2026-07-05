import numpy as np

from src.evaluation import classification_metrics, classify_performance, regression_metrics


def test_classify_performance_uses_thresholds():
    labels = classify_performance(np.array([90, 60, 30, 75, 40]))
    assert list(labels) == ["Excel", "Average", "Struggle", "Excel", "Struggle"]


def test_classification_metrics_returns_key_scores():
    y_true = ["Excel", "Average", "Struggle", "Average"]
    y_pred = ["Excel", "Struggle", "Struggle", "Average"]

    metrics = classification_metrics(y_true, y_pred)

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["balanced_accuracy"] <= 1
    assert metrics["confusion_matrix"].shape == (3, 3)
    assert "per_class" in metrics


def test_regression_metrics_are_computed():
    metrics = regression_metrics([10, 20, 30], [12, 18, 33])

    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
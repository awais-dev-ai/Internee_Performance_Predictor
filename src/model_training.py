"""Training, evaluation, and persistence helpers for intern performance models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.evaluation import classify_performance


@dataclass(frozen=True)
class ModelResult:
    name: str
    model: Any
    predictions: np.ndarray
    metrics: dict[str, float]


# Thresholds for binning performance scores into classes
STRUGGLE_THRESHOLD = 40.0
EXCEL_THRESHOLD = 75.0


def _compute_sample_weights(y: pd.Series) -> np.ndarray:
    """Compute sample weights inversely proportional to class frequency.

    Struggle and Excel samples get higher weight so the model pays more
    attention to minority classes during training.
    """
    labels = classify_performance(y.values)
    class_counts = pd.Series(labels).value_counts()

    # Weight = 1.0 / class_frequency, normalized so mean weight = 1.0
    weights = np.ones(len(y), dtype=float)
    for i, label in enumerate(labels):
        weights[i] = 1.0 / (class_counts.get(label, 1) / len(y))

    # Normalize so mean weight is 1.0 (keeps loss magnitude stable)
    weights = weights / weights.mean()
    return weights


def build_candidate_models(random_state: int = 42) -> dict[str, Any]:
    """Create a compact pair of candidate regressors."""

    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=120,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=140,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            verbosity=0,
        ),
    }


def evaluate_regression_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> ModelResult:
    """Compute regression metrics for a fitted model."""

    predictions = model.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }
    return ModelResult(name=model.__class__.__name__, model=model, predictions=predictions, metrics=metrics)


def train_candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int = 42,
    use_sample_weights: bool = True,
) -> dict[str, Any]:
    """Fit each candidate model on the training set.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    random_state : int
        Random seed for reproducibility.
    use_sample_weights : bool
        If True, compute sample weights inversely proportional to class
        frequency (Struggle/Excel get higher weight) to handle imbalance.
    """
    sample_weights = _compute_sample_weights(y_train) if use_sample_weights else None

    fitted_models: dict[str, Any] = {}
    for name, model in build_candidate_models(random_state=random_state).items():
        if sample_weights is not None:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)
        fitted_models[name] = model
    return fitted_models


def _compute_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute classification metrics from regression predictions."""
    true_labels = classify_performance(y_true.values)
    pred_labels = classify_performance(y_pred)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, pred_labels)),
    }


def select_best_model(
    fitted_models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    alpha: float = 0.5,
) -> ModelResult:
    """Pick the model with the best composite score on the holdout set.

    The composite score balances regression accuracy (RMSE) and
    classification quality (balanced accuracy), ensuring the selected
    model performs well on minority classes.

    Parameters
    ----------
    fitted_models : dict
        Dictionary of model name -> fitted model.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    alpha : float
        Weight for regression component in composite score.
        composite = alpha * (1 - normalized_rmse) + (1 - alpha) * balanced_accuracy
        Higher alpha favors regression accuracy, lower favors classification.
    """
    best_result: ModelResult | None = None
    best_composite: float = -np.inf

    for name, model in fitted_models.items():
        predictions = model.predict(X_test)
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2": float(r2_score(y_test, predictions)),
        }

        # Compute classification metrics
        clf_metrics = _compute_classification_metrics(y_test, predictions)
        metrics["balanced_accuracy"] = clf_metrics["balanced_accuracy"]

        # Normalize RMSE to [0, 1] range for compositing
        # (lower RMSE is better, so we use 1 - normalized)
        y_range = y_test.max() - y_test.min()
        normalized_rmse = metrics["rmse"] / y_range if y_range > 0 else 0.5

        # Composite score: higher is better
        composite = alpha * (1 - normalized_rmse) + (1 - alpha) * clf_metrics["balanced_accuracy"]
        metrics["composite_score"] = float(composite)

        current = ModelResult(name=name, model=model, predictions=predictions, metrics=metrics)
        if composite > best_composite:
            best_composite = composite
            best_result = current

    if best_result is None:
        raise ValueError("No fitted models were provided.")

    return best_result


def build_model_metadata(
    model_name: str,
    metrics: dict[str, float],
    feature_defaults: dict[str, float],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "metrics": metrics,
        "feature_defaults": feature_defaults,
    }


def save_model_artifacts(
    model: Any,
    metadata: dict[str, Any],
    *,
    model_path: str | Path,
    metadata_path: str | Path,
) -> None:
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(metadata, metadata_path)


def load_model_artifacts(
    *,
    model_path: str | Path,
    metadata_path: str | Path,
) -> tuple[Any, dict[str, Any]]:
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    return model, metadata
"""Training, evaluation, and persistence helpers for intern performance models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


@dataclass(frozen=True)
class ModelResult:
	name: str
	model: Any
	predictions: np.ndarray
	metrics: dict[str, float]


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
) -> dict[str, Any]:
	"""Fit each candidate model on the training set."""

	fitted_models: dict[str, Any] = {}
	for name, model in build_candidate_models(random_state=random_state).items():
		model.fit(X_train, y_train)
		fitted_models[name] = model
	return fitted_models


def select_best_model(
	fitted_models: dict[str, Any],
	X_test: pd.DataFrame,
	y_test: pd.Series,
) -> ModelResult:
	"""Pick the model with the lowest RMSE on the holdout set."""

	best_result: ModelResult | None = None

	for name, model in fitted_models.items():
		predictions = model.predict(X_test)
		metrics = {
			"rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
			"mae": float(mean_absolute_error(y_test, predictions)),
			"r2": float(r2_score(y_test, predictions)),
		}
		current = ModelResult(name=name, model=model, predictions=predictions, metrics=metrics)
		if best_result is None or current.metrics["rmse"] < best_result.metrics["rmse"]:
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
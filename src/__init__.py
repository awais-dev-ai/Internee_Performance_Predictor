"""Intern Performance Predictor source package."""

from .data_generation import FEATURE_COLUMNS, TARGET_COLUMN, generate_synthetic_data
from .evaluation import classification_metrics, classify_performance, regression_metrics
from .interpretation import feature_importance_table
from .model_training import (
	build_candidate_models,
	build_model_metadata,
	evaluate_regression_model,
	load_model_artifacts,
	save_model_artifacts,
	select_best_model,
	train_candidate_models,
)
from .preprocessing import (
	CLIP_BOUNDS,
	DatasetSplit,
	prepare_prediction_frame,
	split_features_target,
	train_test_split_data,
	validate_and_clean_dataframe,
)

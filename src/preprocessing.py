"""Functions for data splitting, validation, and cleaning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


FEATURE_COLUMNS = ("task_completion_hrs", "feedback_rating", "attendance_pct")
TARGET_COLUMN = "performance_score"

CLIP_BOUNDS = {
    "task_completion_hrs": (2.0, 20.0),
    "feedback_rating": (1.0, 5.0),
    "attendance_pct": (50.0, 100.0),
    TARGET_COLUMN: (0.0, 100.0),
}

# Thresholds for binning the continuous target into stratified groups
STRATIFY_THRESHOLDS = (40.0, 75.0)
STRATIFY_LABELS = [0, 1, 2]  # 0=Struggle, 1=Average, 2=Excel


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def _required_columns(include_target: bool = True) -> tuple[str, ...]:
    if include_target:
        return FEATURE_COLUMNS + (TARGET_COLUMN,)
    return FEATURE_COLUMNS


def _bin_for_stratification(y: pd.Series) -> np.ndarray:
    """Bin continuous target into strata for stratified splitting."""
    bins = [-np.inf] + list(STRATIFY_THRESHOLDS) + [np.inf]
    return np.digitize(y, bins=bins) - 1  # Returns 0, 1, or 2


def validate_and_clean_dataframe(
    df: pd.DataFrame,
    *,
    include_target: bool = True,
    fill_values: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Validate required columns, coerce numeric values, and clean noisy data."""

    required_columns = _required_columns(include_target=include_target)
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    cleaned = df.copy()
    numeric_columns = list(required_columns)

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)

    defaults = fill_values or {}
    for column in numeric_columns:
        if cleaned[column].isna().any():
            fallback = defaults.get(column)
            if fallback is None or pd.isna(fallback):
                fallback = cleaned[column].median()
            if pd.isna(fallback):
                fallback = CLIP_BOUNDS[column][0]
            cleaned[column] = cleaned[column].fillna(fallback)

        lower, upper = CLIP_BOUNDS[column]
        cleaned[column] = cleaned[column].clip(lower, upper)

    return cleaned


def split_features_target(
    df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix and target series after cleaning the dataframe."""

    cleaned = validate_and_clean_dataframe(df, include_target=True)
    features = cleaned[list(FEATURE_COLUMNS)].copy()
    target = cleaned[target_column].copy()
    return features, target


def train_test_split_data(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> DatasetSplit:
    """Clean the dataframe and create a reproducible train/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with features and target.
    test_size : float
        Proportion of data to use for testing (default 0.2).
    random_state : int
        Random seed for reproducibility.
    stratify : bool
        If True, use stratified splitting based on binned performance categories.
        This preserves the 15/70/15 class distribution in both train and test sets.
    """
    features, target = split_features_target(df)

    if stratify:
        # Bin the continuous target into strata for stratified splitting
        strata = _bin_for_stratification(target)

        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, test_idx = next(splitter.split(features, strata))

        X_train = features.iloc[train_idx].reset_index(drop=True)
        X_test = features.iloc[test_idx].reset_index(drop=True)
        y_train = target.iloc[train_idx].reset_index(drop=True)
        y_test = target.iloc[test_idx].reset_index(drop=True)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
        )

    return DatasetSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def prepare_prediction_frame(
    payload: dict[str, float] | pd.DataFrame,
    *,
    fill_values: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Validate a single prediction payload from the UI or API."""

    if isinstance(payload, pd.DataFrame):
        frame = payload.copy()
    else:
        frame = pd.DataFrame([payload])

    cleaned = validate_and_clean_dataframe(
        frame,
        include_target=False,
        fill_values=fill_values,
    )
    return cleaned[list(FEATURE_COLUMNS)].copy()
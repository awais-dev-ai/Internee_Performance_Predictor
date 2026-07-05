import numpy as np
import pandas as pd
import pytest

from src.data_generation import generate_synthetic_data
from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, prepare_prediction_frame, split_features_target, validate_and_clean_dataframe


def test_validate_and_clean_dataframe_imputes_and_clips():
    df = generate_synthetic_data(n_samples=30, seed=21)
    df.loc[0, "feedback_rating"] = np.nan
    df.loc[1, "attendance_pct"] = 999
    df.loc[2, "task_completion_hrs"] = -5

    cleaned = validate_and_clean_dataframe(df)

    assert not cleaned.isna().any().any()
    assert cleaned["feedback_rating"].between(1, 5).all()
    assert cleaned["attendance_pct"].between(50, 100).all()
    assert cleaned["task_completion_hrs"].between(2, 20).all()


def test_validate_and_clean_dataframe_missing_column_raises():
    df = pd.DataFrame({"task_completion_hrs": [1.0], "feedback_rating": [2.0]})

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_and_clean_dataframe(df)


def test_split_features_target_returns_expected_shapes():
    df = generate_synthetic_data(n_samples=12, seed=4)
    X, y = split_features_target(df)

    assert list(X.columns) == list(FEATURE_COLUMNS)
    assert y.name == TARGET_COLUMN
    assert X.shape == (12, 3)
    assert y.shape == (12,)


def test_prepare_prediction_frame_fills_missing_inputs():
    payload = {
        "task_completion_hrs": "",
        "feedback_rating": "4.2",
        "attendance_pct": "110",
    }
    frame = prepare_prediction_frame(
        payload,
        fill_values={
            "task_completion_hrs": 8.0,
            "feedback_rating": 3.5,
            "attendance_pct": 85.0,
        },
    )

    assert list(frame.columns) == list(FEATURE_COLUMNS)
    assert frame.shape == (1, 3)
    assert frame.iloc[0]["task_completion_hrs"] == 8.0
    assert frame.iloc[0]["attendance_pct"] == 100.0
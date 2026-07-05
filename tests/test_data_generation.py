from src.data_generation import FEATURE_COLUMNS, TARGET_COLUMN, generate_synthetic_data, inject_missing_values


def test_generate_synthetic_data_structure(tmp_path):
    df = generate_synthetic_data(n_samples=50, seed=11, save_path=tmp_path / "interns.csv")

    assert df.shape == (50, 5)
    assert list(FEATURE_COLUMNS) == ["task_completion_hrs", "feedback_rating", "attendance_pct"]
    assert TARGET_COLUMN in df.columns
    assert (tmp_path / "interns.csv").exists()
    assert df["task_completion_hrs"].between(2, 20).all()
    assert df["feedback_rating"].between(1, 5).all()
    assert df["attendance_pct"].between(50, 100).all()
    assert df[TARGET_COLUMN].between(0, 100).all()


def test_inject_missing_values_adds_nans():
    df = generate_synthetic_data(n_samples=20, seed=3)
    corrupted = inject_missing_values(df, missing_rate=0.3, seed=99)

    assert corrupted.isna().sum().sum() > 0
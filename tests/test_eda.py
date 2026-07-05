from src.data_generation import generate_synthetic_data
from src.eda import correlation_matrix, data_overview


def test_data_overview_reports_missing_values():
    df = generate_synthetic_data(n_samples=25, seed=41)
    overview = data_overview(df)

    assert overview["shape"] == (25, 5)
    assert overview["missing_values"] == 0
    assert "performance_score" in overview["columns"]


def test_correlation_matrix_is_square():
    df = generate_synthetic_data(n_samples=25, seed=41)
    corr = correlation_matrix(df)

    assert corr.shape[0] == corr.shape[1]
    assert "task_completion_hrs" in corr.columns
from src.data_generation import FEATURE_COLUMNS, generate_synthetic_data
from src.interpretation import feature_importance_table
from src.model_training import train_candidate_models
from src.preprocessing import train_test_split_data


def test_feature_importance_table_returns_sorted_rows():
    df = generate_synthetic_data(n_samples=140, seed=33)
    split = train_test_split_data(df, test_size=0.25, random_state=33)
    fitted_models = train_candidate_models(split.X_train, split.y_train, random_state=33)
    model = fitted_models["Random Forest"]

    table = feature_importance_table(model, list(FEATURE_COLUMNS))

    assert list(table.columns) == ["feature", "importance"]
    assert table.shape == (3, 2)
    assert table["importance"].is_monotonic_decreasing
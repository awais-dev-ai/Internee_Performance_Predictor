from src.data_generation import generate_synthetic_data
from src.model_training import evaluate_regression_model, load_model_artifacts, save_model_artifacts, select_best_model, train_candidate_models
from src.preprocessing import train_test_split_data


def test_training_selects_a_reasonable_model():
    df = generate_synthetic_data(n_samples=220, seed=15)
    split = train_test_split_data(df, test_size=0.25, random_state=15)
    fitted_models = train_candidate_models(split.X_train, split.y_train, random_state=15)
    best_result = select_best_model(fitted_models, split.X_test, split.y_test)

    assert best_result.name in fitted_models
    assert best_result.metrics["rmse"] < 12
    assert best_result.metrics["r2"] > 0.75


def test_save_and_load_model_artifacts(tmp_path):
    df = generate_synthetic_data(n_samples=120, seed=18)
    split = train_test_split_data(df, test_size=0.25, random_state=18)
    fitted_models = train_candidate_models(split.X_train, split.y_train, random_state=18)
    best_result = select_best_model(fitted_models, split.X_test, split.y_test)
    metadata = {"model_name": best_result.name, "metrics": best_result.metrics, "feature_defaults": split.X_train.median().to_dict()}

    model_path = tmp_path / "model.pkl"
    metadata_path = tmp_path / "metadata.pkl"
    save_model_artifacts(best_result.model, metadata, model_path=model_path, metadata_path=metadata_path)

    loaded_model, loaded_metadata = load_model_artifacts(model_path=model_path, metadata_path=metadata_path)
    assert loaded_metadata["model_name"] == best_result.name
    assert loaded_model.predict(split.X_test[:3]).shape == (3,)
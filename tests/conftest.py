from __future__ import annotations

from pathlib import Path

import pytest

from src.data_generation import generate_synthetic_data
from src.model_training import build_model_metadata, save_model_artifacts, select_best_model, train_candidate_models
from src.preprocessing import train_test_split_data


@pytest.fixture(scope="session")
def synthetic_dataset():
    return generate_synthetic_data(n_samples=240, seed=123)


@pytest.fixture()
def dataset_split(synthetic_dataset):
    return train_test_split_data(synthetic_dataset, test_size=0.25, random_state=7)


@pytest.fixture()
def trained_bundle(dataset_split):
    fitted_models = train_candidate_models(dataset_split.X_train, dataset_split.y_train, random_state=7)
    best_result = select_best_model(fitted_models, dataset_split.X_test, dataset_split.y_test)
    metadata = build_model_metadata(
        model_name=best_result.name,
        metrics=best_result.metrics,
        feature_defaults=dataset_split.X_train.median().to_dict(),
    )
    return {
        "best_result": best_result,
        "metadata": metadata,
        "dataset_split": dataset_split,
    }


@pytest.fixture()
def saved_artifacts(tmp_path: Path, trained_bundle):
    model_path = tmp_path / "models" / "best_model.pkl"
    metadata_path = tmp_path / "models" / "model_metadata.pkl"
    save_model_artifacts(
        trained_bundle["best_result"].model,
        trained_bundle["metadata"],
        model_path=model_path,
        metadata_path=metadata_path,
    )
    return {
        "base_dir": tmp_path,
        "model_path": model_path,
        "metadata_path": metadata_path,
    }
"""Main pipeline script for training and saving the intern performance model."""

from __future__ import annotations

from pathlib import Path

from src.data_generation import generate_synthetic_data
from src.model_training import (
    build_model_metadata,
    save_model_artifacts,
    select_best_model,
    train_candidate_models,
)
from src.preprocessing import train_test_split_data


def run_pipeline() -> dict[str, object]:
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "intern_performance_data.csv"
    model_path = project_root / "models" / "best_model.pkl"
    metadata_path = project_root / "models" / "model_metadata.pkl"

    dataset = generate_synthetic_data(save_path=data_path)
    split = train_test_split_data(dataset)
    fitted_models = train_candidate_models(split.X_train, split.y_train)
    best_result = select_best_model(fitted_models, split.X_test, split.y_test)
    feature_defaults = split.X_train.median().to_dict()
    metadata = build_model_metadata(
        model_name=best_result.name,
        metrics=best_result.metrics,
        feature_defaults=feature_defaults,
    )
    save_model_artifacts(
        best_result.model,
        metadata,
        model_path=model_path,
        metadata_path=metadata_path,
    )

    return {
        "model_name": best_result.name,
        "metrics": best_result.metrics,
        "data_path": str(data_path),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
    }


if __name__ == "__main__":
    result = run_pipeline()
    print("Pipeline complete")
    print(result)
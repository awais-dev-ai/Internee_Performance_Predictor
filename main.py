"""Main pipeline script for training and saving the intern performance model."""

from __future__ import annotations

from pathlib import Path

from src.data_generation import generate_synthetic_data
from src.evaluation import optimize_thresholds
from src.model_training import (
    build_model_metadata,
    save_model_artifacts,
    select_best_model,
    train_candidate_models,
)
from src.preprocessing import train_test_split_data


def run_pipeline(
    *,
    n_samples: int = 2000,
    use_sample_weights: bool = True,
    stratify_split: bool = True,
    optimize_classification_thresholds: bool = True,
) -> dict[str, object]:
    """Run the full training pipeline with class imbalance handling.

    Parameters
    ----------
    n_samples : int
        Number of synthetic samples to generate.
    use_sample_weights : bool
        If True, use sample weights during training to handle class imbalance.
    stratify_split : bool
        If True, use stratified train/test split to preserve class proportions.
    optimize_classification_thresholds : bool
        If True, grid search for optimal classification thresholds.
    """
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "intern_performance_data.csv"
    model_path = project_root / "models" / "best_model.pkl"
    metadata_path = project_root / "models" / "model_metadata.pkl"

    dataset = generate_synthetic_data(
        n_samples=n_samples,
        save_path=data_path,
        oversample_minority=True,
    )
    split = train_test_split_data(dataset, stratify=stratify_split)
    fitted_models = train_candidate_models(
        split.X_train, split.y_train,
        use_sample_weights=use_sample_weights,
    )
    best_result = select_best_model(fitted_models, split.X_test, split.y_test)

    # Optimize classification thresholds if requested
    threshold_optimization = None
    if optimize_classification_thresholds:
        threshold_optimization = optimize_thresholds(
            split.y_test.values,
            best_result.predictions,
            metric="macro_f1",
        )

    feature_defaults = split.X_train.median().to_dict()
    metadata = build_model_metadata(
        model_name=best_result.name,
        metrics=best_result.metrics,
        feature_defaults=feature_defaults,
    )

    # Include threshold optimization results in metadata
    if threshold_optimization is not None:
        metadata["threshold_optimization"] = {
            "struggle_threshold": threshold_optimization["struggle_threshold"],
            "excel_threshold": threshold_optimization["excel_threshold"],
            "best_macro_f1": threshold_optimization["best_score"],
        }

    save_model_artifacts(
        best_result.model,
        metadata,
        model_path=model_path,
        metadata_path=metadata_path,
    )

    return {
        "model_name": best_result.name,
        "metrics": best_result.metrics,
        "threshold_optimization": threshold_optimization,
        "data_path": str(data_path),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
    }


if __name__ == "__main__":
    result = run_pipeline()
    print("Pipeline complete")
    print(f"Model: {result['model_name']}")
    print(f"Metrics: RMSE={result['metrics']['rmse']:.2f}, "
          f"MAE={result['metrics']['mae']:.2f}, "
          f"R²={result['metrics']['r2']:.3f}, "
          f"Balanced Acc={result['metrics'].get('balanced_accuracy', 'N/A')}")
    if result.get("threshold_optimization"):
        topt = result["threshold_optimization"]
        print(f"Optimal thresholds: Struggle ≤ {topt['struggle_threshold']:.0f}, "
              f"Excel ≥ {topt['excel_threshold']:.0f} "
              f"(Macro F1: {topt['best_score']:.3f})")
    print(f"Data → {result['data_path']}")
    print(f"Model → {result['model_path']}")
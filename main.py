"""Main pipeline script for training and saving the intern performance model."""

from __future__ import annotations

from pathlib import Path

from src.data_generation import generate_synthetic_data
from src.evaluation import optimize_thresholds, regression_metrics
from src.model_training import (
    build_model_metadata,
    save_model_artifacts,
    select_best_model,
    tune_candidate_models,           # <-- automated tuning
)
from src.preprocessing import train_val_test_split   # <-- validation split


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

    # Split into train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        dataset,
        val_size=0.15,
        test_size=0.2,
        random_state=42,
        stratify=stratify_split,
    )

    # Train with automated hyperparameter tuning (reproducible)
    fitted_models = tune_candidate_models(
        X_train, y_train,
        use_sample_weights=use_sample_weights,
        random_state=42,              # <-- added for reproducibility
    )

    # Select best model based on validation set (not test)
    best_result = select_best_model(fitted_models, X_val, y_val)

    # Optimize thresholds on validation set predictions
    threshold_optimization = None
    if optimize_classification_thresholds:
        threshold_optimization = optimize_thresholds(
            y_val.values,
            best_result.predictions,
            metric="macro_f1",
        )

    # Final evaluation on test set for reporting (unbiased)
    final_preds = best_result.model.predict(X_test)
    test_metrics = regression_metrics(y_test, final_preds)

    # Build metadata
    feature_defaults = X_train.median().to_dict()
    metadata = build_model_metadata(
        model_name=best_result.name,
        metrics=best_result.metrics,   # these are from validation
        feature_defaults=feature_defaults,
    )

    # Include threshold optimization results in metadata
    if threshold_optimization is not None:
        metadata["threshold_optimization"] = {
            "struggle_threshold": threshold_optimization["struggle_threshold"],
            "excel_threshold": threshold_optimization["excel_threshold"],
            "best_macro_f1": threshold_optimization["best_score"],
        }

    # Store test metrics for reference (not used for model selection)
    metadata["test_metrics"] = test_metrics

    save_model_artifacts(
        best_result.model,
        metadata,
        model_path=model_path,
        metadata_path=metadata_path,
    )

    return {
        "model_name": best_result.name,
        "metrics": best_result.metrics,
        "test_metrics": test_metrics,
        "threshold_optimization": threshold_optimization,
        "data_path": str(data_path),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000,
                        help="Number of synthetic samples to generate")
    args = parser.parse_args()

    result = run_pipeline(n_samples=args.samples)
    print("Pipeline complete")
    print(f"Model: {result['model_name']}")
    print(f"Validation Metrics: RMSE={result['metrics']['rmse']:.2f}, "
          f"MAE={result['metrics']['mae']:.2f}, "
          f"R²={result['metrics']['r2']:.3f}, "
          f"Balanced Acc={result['metrics'].get('balanced_accuracy', 'N/A')}")
    if result.get("test_metrics"):
        tm = result["test_metrics"]
        print(f"Test Metrics: RMSE={tm['rmse']:.2f}, MAE={tm['mae']:.2f}, R²={tm['r2']:.3f}")
    if result.get("threshold_optimization"):
        topt = result["threshold_optimization"]
        print(f"Optimal thresholds: Struggle ≤ {topt['struggle_threshold']:.0f}, "
              f"Excel ≥ {topt['excel_threshold']:.0f} "
              f"(Macro F1: {topt['best_score']:.3f})")
    print(f"Data → {result['data_path']}")
    print(f"Model → {result['model_path']}")

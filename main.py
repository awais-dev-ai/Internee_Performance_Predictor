"""
Main pipeline script for training and saving the intern performance model.

This script:
1. Generates synthetic data (or reads real data).
2. Trains and tunes multiple models (XGBoost, Random Forest).
3. Selects the best model based on validation performance.
4. Saves the model and metadata locally.
5. Automatically uploads the model to Hugging Face Hub if HF_TOKEN is set.

Usage:
    python main.py                # Train with default 2000 samples
    python main.py --samples 500  # Train with 500 samples (fast validation)
"""

from __future__ import annotations

import os
from pathlib import Path

# --- OFFICIAL FIX: Suppress sklearn parallel warning (documented environment variable) ---
os.environ["SKLEARN_IGNORE_PARALLEL_WARNING"] = "1"

# --- Import pipeline components from our src/ package ---
from src.data_generation import generate_synthetic_data
from src.evaluation import classification_metrics, classify_performance, optimize_thresholds, regression_metrics
from src.model_training import (
    build_model_metadata,
    oversample_training_data,
    save_metadata_json,
    save_model_artifacts,
    select_best_model,
    tune_candidate_models,
)
from src.preprocessing import train_val_test_split

# --- Optional Hugging Face Hub import ---
try:
    from huggingface_hub import HfApi, upload_file
except ImportError:
    HfApi = None
    upload_file = None

# --- Environment Configuration ---
from dotenv import load_dotenv
load_dotenv()  # Reads .env file and adds variables to os.environ


def upload_model_to_hub(
    model_path: Path,
    metadata_path: Path,
    token: str | None = None,
    repo_id: str = "awais-dev-ai/Intern-Performance-Model",
) -> None:
    """
    Upload the trained model and metadata to Hugging Face Hub.

    This function is called automatically at the end of the training pipeline
    if HF_TOKEN is available in the environment.

    Parameters
    ----------
    model_path : Path
        Path to the saved model file (best_model.pkl).
    metadata_path : Path
        Path to the saved metadata JSON file (model_metadata.json).
    token : str, optional
        Hugging Face access token. If not provided, reads from HF_TOKEN env var.
    repo_id : str
        Hugging Face repository ID (your model repo name).
    """
    if HfApi is None or upload_file is None:
        print("⚠️  huggingface_hub not installed. Skipping upload.")
        return

    if token is None:
        token = os.environ.get("HF_TOKEN")
    if token is None:
        print("⚠️  HF_TOKEN not set. Skipping upload.")
        return

    try:
        if model_path.exists():
            upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo="best_model.pkl",
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )
            print(f"✅ Model uploaded to {repo_id}")
        else:
            print(f"⚠️  Model file not found: {model_path}")

        if metadata_path.exists():
            upload_file(
                path_or_fileobj=str(metadata_path),
                path_in_repo="model_metadata.json",
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )
            print(f"✅ Metadata uploaded to {repo_id}")
        else:
            print(f"⚠️  Metadata file not found: {metadata_path}")

    except Exception as e:
        print(f"❌ Upload failed: {e}")


def run_pipeline(
    *,
    n_samples: int = 2000,
    use_sample_weights: bool = True,
    stratify_split: bool = True,
    optimize_classification_thresholds: bool = True,
) -> dict[str, object]:
    """
    Run the full training pipeline with class imbalance handling.

    This is the main orchestrator function that:
        - Generates data
        - Splits into train/validation/test
        - Oversamples the training fold to handle imbalance
        - Trains and tunes models
        - Selects the best model
        - Optimizes classification thresholds
        - Evaluates on the test set
        - Saves model artifacts locally
        - Uploads to Hugging Face Hub (if token is set)

    Returns
    -------
    dict[str, object]
        A dictionary containing model name, metrics, paths, and optimization results.
    """
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "intern_performance_data.csv"
    model_path = project_root / "models" / "best_model.pkl"
    metadata_path = project_root / "models" / "model_metadata.pkl"

    dataset = generate_synthetic_data(
        n_samples=n_samples,
        save_path=data_path,
        oversample_minority=False,
    )

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        dataset,
        val_size=0.15,
        test_size=0.2,
        random_state=42,
        stratify=stratify_split,
    )

    X_train, y_train = oversample_training_data(
        X_train, y_train,
        jitter_scale=0.05,
        random_state=42,
    )

    fitted_models = tune_candidate_models(
        X_train, y_train,
        use_sample_weights=use_sample_weights,
        random_state=42,
    )

    best_result = select_best_model(fitted_models, X_val, y_val)

    threshold_optimization = None
    if optimize_classification_thresholds:
        threshold_optimization = optimize_thresholds(
            y_val.values,
            best_result.predictions,
            metric="macro_f1",
        )

    final_preds = best_result.model.predict(X_test)
    test_metrics = regression_metrics(y_test, final_preds)

    struggle_th = (threshold_optimization or {}).get("struggle_threshold", 40.0)
    excel_th = (threshold_optimization or {}).get("excel_threshold", 75.0)
    true_labels = classify_performance(y_test.values, struggle_threshold=struggle_th, excel_threshold=excel_th)
    pred_labels = classify_performance(final_preds, struggle_threshold=struggle_th, excel_threshold=excel_th)
    clf_report = classification_metrics(true_labels, pred_labels)

    feature_defaults = X_train.median().to_dict()
    metadata = build_model_metadata(
        model_name=best_result.name,
        metrics=best_result.metrics,
        feature_defaults=feature_defaults,
        classification=clf_report,
    )

    if threshold_optimization is not None:
        metadata["threshold_optimization"] = {
            "struggle_threshold": threshold_optimization["struggle_threshold"],
            "excel_threshold": threshold_optimization["excel_threshold"],
            "best_macro_f1": threshold_optimization["best_score"],
        }

    metadata["test_metrics"] = test_metrics

    save_model_artifacts(
        best_result.model,
        metadata,
        model_path=model_path,
        metadata_path=metadata_path,
    )

    json_path = project_root / "models" / "model_metadata.json"
    save_metadata_json(metadata, json_path)

    upload_model_to_hub(model_path, json_path)

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

    parser = argparse.ArgumentParser(
        description="Train the intern performance prediction model."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of synthetic samples to generate (default: 2000). "
             "Use 500 for fast CI/CD validation.",
    )
    args = parser.parse_args()

    result = run_pipeline(n_samples=args.samples)

    # --- Print results ---
    print("\n" + "=" * 50)
    print("🚀 PIPELINE COMPLETE")
    print("=" * 50)
    print(f"✅ Best Model: {result['model_name']}")
    print("\n📊 Validation Metrics:")
    print(f"   RMSE: {result['metrics']['rmse']:.2f}")
    print(f"   MAE:  {result['metrics']['mae']:.2f}")
    print(f"   R²:   {result['metrics']['r2']:.3f}")
    print(f"   Balanced Acc: {result['metrics'].get('balanced_accuracy', 'N/A')}")

    if result.get("test_metrics"):
        tm = result["test_metrics"]
        print("\n📊 Test Metrics (Unbiased):")
        print(f"   RMSE: {tm['rmse']:.2f}")
        print(f"   MAE:  {tm['mae']:.2f}")
        print(f"   R²:   {tm['r2']:.3f}")

    if result.get("threshold_optimization"):
        topt = result["threshold_optimization"]
        print("\n🎯 Optimized Thresholds:")
        print(f"   Struggle ≤ {topt['struggle_threshold']:.0f}")
        print(f"   Excel ≥ {topt['excel_threshold']:.0f}")
        print(f"   Macro F1: {topt['best_score']:.3f}")

    print(f"\n📁 Data saved to: {result['data_path']}")
    print(f"📁 Model saved to: {result['model_path']}")
    print("=" * 50)
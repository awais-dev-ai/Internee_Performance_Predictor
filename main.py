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

from __future__ import annotations  # Allows using modern type hints (e.g., dict[str, object])

import os
from pathlib import Path

# --- Environment Configuration ---
# Load environment variables from a .env file (local development).
# This allows us to store HF_TOKEN securely without hardcoding it.
# In CI/CD (GitHub Actions), the token is injected directly via secrets.
from dotenv import load_dotenv
load_dotenv()  # Reads .env file and adds variables to os.environ

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
# Gracefully handle missing installation. If the user doesn't have huggingface_hub
# installed, the script will skip uploading but still train and save locally.
try:
    from huggingface_hub import HfApi, upload_file
except ImportError:
    HfApi = None


def upload_model_to_hub(
    model_path: Path,
    metadata_path: Path,
    token: str | None = None,
    repo_id: str = "awais-dev-ai/Intern-Performance-Predictor",
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
        Hugging Face repository ID (your Space name).
    """
    # Check if the huggingface_hub library is installed
    if HfApi is None:
        print("⚠️  huggingface_hub not installed. Skipping upload.")
        return

    # If token wasn't passed directly, try to read it from environment variables
    if token is None:
        token = os.environ.get("HF_TOKEN")
    if token is None:
        print("⚠️  HF_TOKEN not set. Skipping upload.")
        return

    try:
        # Initialize the Hugging Face API client
        api = HfApi()

        # --- Upload the model file (.pkl) ---
        if model_path.exists():
            upload_file(
                path_or_fileobj=str(model_path),          # Local file path
                path_in_repo="models/best_model.pkl",     # Destination path on HF Hub
                repo_id=repo_id,                          # Your Space ID
                repo_type="space",                        # It's a Space, not a model repo
                token=token,
            )
            print(f"✅ Model uploaded to {repo_id}")
        else:
            print(f"⚠️  Model file not found: {model_path}")

        # --- Upload the metadata file (.json) ---
        if metadata_path.exists():
            upload_file(
                path_or_fileobj=str(metadata_path),
                path_in_repo="models/model_metadata.json",
                repo_id=repo_id,
                repo_type="space",
                token=token,
            )
            print(f"✅ Metadata uploaded to {repo_id}")
        else:
            print(f"⚠️  Metadata file not found: {metadata_path}")

    except Exception as e:
        # Catch any upload errors (network issues, auth failures, etc.)
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
    # --- 1. Setup paths ---
    # Path(__file__).resolve().parent gives the absolute path to the folder
    # containing main.py. This ensures the script works regardless of where you run it from.
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "intern_performance_data.csv"
    model_path = project_root / "models" / "best_model.pkl"
    metadata_path = project_root / "models" / "model_metadata.pkl"

    # --- 2. Generate synthetic dataset ---
    # If you have real data, replace this with a read from CSV.
    dataset = generate_synthetic_data(
        n_samples=n_samples,
        save_path=data_path,
        oversample_minority=False,  # Oversampling is done AFTER split to avoid data leakage
    )

    # --- 3. Split data into train, validation, and test sets ---
    # Stratify ensures that the rare classes (Struggle/Excel) are proportionally
    # represented in all splits.
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        dataset,
        val_size=0.15,
        test_size=0.2,
        random_state=42,
        stratify=stratify_split,
    )

    # --- 4. Oversample ONLY the training fold ---
    # This prevents data leakage. We apply SMOTE/jitter ONLY on the training set
    # after splitting, so test/validation sets remain untouched and honest.
    X_train, y_train = oversample_training_data(
        X_train, y_train,
        jitter_scale=0.05,
        random_state=42,
    )

    # --- 5. Train and tune candidate models ---
    # This runs GridSearchCV over Random Forest and XGBoost with class weights.
    fitted_models = tune_candidate_models(
        X_train, y_train,
        use_sample_weights=use_sample_weights,
        random_state=42,
    )

    # --- 6. Select the best model using validation set ---
    # We choose the model with the best composite score (RMSE + Balanced Accuracy)
    # on the validation set, NOT the test set.
    best_result = select_best_model(fitted_models, X_val, y_val)

    # --- 7. Optimize classification thresholds ---
    # Grid search over struggle/excel cutoffs to maximize macro F1 on validation.
    threshold_optimization = None
    if optimize_classification_thresholds:
        threshold_optimization = optimize_thresholds(
            y_val.values,
            best_result.predictions,
            metric="macro_f1",
        )

    # --- 8. Final evaluation on the test set ---
    # This gives us an unbiased, honest performance estimate.
    final_preds = best_result.model.predict(X_test)
    test_metrics = regression_metrics(y_test, final_preds)

    # --- 9. Compute classification metrics with optimized thresholds ---
    struggle_th = (threshold_optimization or {}).get("struggle_threshold", 40.0)
    excel_th = (threshold_optimization or {}).get("excel_threshold", 75.0)
    true_labels = classify_performance(y_test.values, struggle_threshold=struggle_th, excel_threshold=excel_th)
    pred_labels = classify_performance(final_preds, struggle_threshold=struggle_th, excel_threshold=excel_th)
    clf_report = classification_metrics(true_labels, pred_labels)

    # --- 10. Build metadata dictionary ---
    # This contains model name, validation metrics, feature defaults, thresholds, etc.
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

    # --- 11. Save model and metadata to disk (locally) ---
    # This creates models/best_model.pkl and models/model_metadata.pkl
    save_model_artifacts(
        best_result.model,
        metadata,
        model_path=model_path,
        metadata_path=metadata_path,
    )

    # --- 12. Save human-readable JSON version of metadata ---
    # This is used by the web app and the README updater script.
    json_path = project_root / "models" / "model_metadata.json"
    save_metadata_json(metadata, json_path)

    # --- 13. Automatically upload to Hugging Face Hub ---
    # If HF_TOKEN is set (either in .env or in the environment), upload the
    # model and metadata to your Space. This makes the model available for
    # the Space to download on startup.
    upload_model_to_hub(model_path, json_path)

    # Return results for CLI printing or further use
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
    """
    Command-line entry point.

    Usage:
        python main.py              # Full training (2000 samples)
        python main.py --samples 500  # Fast validation (500 samples)
    """
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

    # Run the full pipeline with the specified number of samples
    result = run_pipeline(n_samples=args.samples)

    # --- Print results to the console ---
    print("\n" + "=" * 50)
    print("🚀 PIPELINE COMPLETE")
    print("=" * 50)
    print(f"✅ Best Model: {result['model_name']}")
    print(f"\n📊 Validation Metrics:")
    print(f"   RMSE: {result['metrics']['rmse']:.2f}")
    print(f"   MAE:  {result['metrics']['mae']:.2f}")
    print(f"   R²:   {result['metrics']['r2']:.3f}")
    print(f"   Balanced Acc: {result['metrics'].get('balanced_accuracy', 'N/A')}")

    if result.get("test_metrics"):
        tm = result["test_metrics"]
        print(f"\n📊 Test Metrics (Unbiased):")
        print(f"   RMSE: {tm['rmse']:.2f}")
        print(f"   MAE:  {tm['mae']:.2f}")
        print(f"   R²:   {tm['r2']:.3f}")

    if result.get("threshold_optimization"):
        topt = result["threshold_optimization"]
        print(f"\n🎯 Optimized Thresholds:")
        print(f"   Struggle ≤ {topt['struggle_threshold']:.0f}")
        print(f"   Excel ≥ {topt['excel_threshold']:.0f}")
        print(f"   Macro F1: {topt['best_score']:.3f}")

    print(f"\n📁 Data saved to: {result['data_path']}")
    print(f"📁 Model saved to: {result['model_path']}")
    print("=" * 50)
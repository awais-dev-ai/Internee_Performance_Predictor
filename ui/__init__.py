"""Flask application factory for the intern performance predictor UI."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template, request

from src.data_generation import generate_synthetic_data
from src.evaluation import classify_performance, optimize_thresholds
from src.model_training import (
    build_model_metadata,
    load_model_artifacts,
    save_model_artifacts,
    select_best_model,
    train_candidate_models,
)
from src.preprocessing import FEATURE_COLUMNS, prepare_prediction_frame, train_test_split_data


def _project_paths(base_dir: Path | None = None) -> dict[str, Path]:
    """Resolve project paths relative to the UI package root."""

    project_root = base_dir or Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    model_dir = project_root / "models"
    return {
        "root": project_root,
        "data_dir": data_dir,
        "model_dir": model_dir,
        "data_path": data_dir / "intern_performance_data.csv",
        "model_path": model_dir / "best_model.pkl",
        "metadata_path": model_dir / "model_metadata.pkl",
    }


def ensure_artifacts(paths: dict[str, Path]) -> tuple[object, dict]:
    """Load saved artifacts or train a fresh model if none exist.

    Uses stratified splitting, sample weights, and threshold optimization
    to handle class imbalance for Struggle and Excel predictions.
    """

    if paths["model_path"].exists() and paths["metadata_path"].exists():
        return load_model_artifacts(
            model_path=paths["model_path"],
            metadata_path=paths["metadata_path"],
        )

    paths["data_dir"].mkdir(parents=True, exist_ok=True)
    paths["model_dir"].mkdir(parents=True, exist_ok=True)

    # Generate data with oversampling for minority classes
    dataset = generate_synthetic_data(
        n_samples=2000,
        save_path=paths["data_path"],
        oversample_minority=True,
    )

    # Stratified split preserves 15/70/15 distribution
    split = train_test_split_data(dataset, stratify=True)

    # Train with sample weights to handle imbalance
    fitted_models = train_candidate_models(
        split.X_train, split.y_train,
        use_sample_weights=True,
    )

    # Select best model using composite score (regression + classification)
    best_result = select_best_model(fitted_models, split.X_test, split.y_test)

    # Optimize classification thresholds
    threshold_opt = optimize_thresholds(
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

    # Store optimized thresholds in metadata
    metadata["threshold_optimization"] = {
        "struggle_threshold": threshold_opt["struggle_threshold"],
        "excel_threshold": threshold_opt["excel_threshold"],
        "best_macro_f1": threshold_opt["best_score"],
    }

    save_model_artifacts(
        best_result.model,
        metadata,
        model_path=paths["model_path"],
        metadata_path=paths["metadata_path"],
    )
    return best_result.model, metadata


def create_app(
    *,
    base_dir: Path | None = None,
    data_path: str | Path | None = None,
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> Flask:
    """Create and configure the Flask application."""

    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parent / "templates"),
        static_folder=str(Path(__file__).resolve().parent / "static"),
    )

    paths = _project_paths(base_dir)
    if data_path is not None:
        paths["data_path"] = Path(data_path)
    if model_path is not None:
        paths["model_path"] = Path(model_path)
    if metadata_path is not None:
        paths["metadata_path"] = Path(metadata_path)

    model, metadata = ensure_artifacts(paths)
    defaults = metadata.get("feature_defaults", {})

    # Use optimized thresholds if available, otherwise use defaults
    threshold_opt = metadata.get("threshold_optimization", {})
    struggle_threshold = threshold_opt.get("struggle_threshold", 40.0)
    excel_threshold = threshold_opt.get("excel_threshold", 75.0)

    def predict_from_payload(payload: dict[str, object]) -> dict[str, object]:
        """Run the full prediction pipeline on a raw form payload."""

        feature_frame = prepare_prediction_frame(payload, fill_values=defaults)
        predicted_score = float(model.predict(feature_frame)[0])
        category = classify_performance(
            [predicted_score],
            struggle_threshold=struggle_threshold,
            excel_threshold=excel_threshold,
        )[0]

        # Compute feature contributions
        features_list = []
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = list(FEATURE_COLUMNS)
            input_values = feature_frame.iloc[0]
            # Get median values as baseline
            baseline = feature_frame.median()

            for i, name in enumerate(feature_names):
                if i < len(importances):
                    # How far from baseline, scaled by importance
                    deviation = float(input_values[name] - baseline[name])
                    # Get max deviation in training to normalize
                    direction = "positive" if deviation >= 0 else "negative"
                    # Scale contribution percentage
                    abs_dev = abs(deviation)
                    max_dev = max(abs(float(input_values[name])), 1.0)
                    pct = min(abs_dev / max_dev * 100 * float(importances[i]), 100)
                    features_list.append({
                        "name": name,
                        "value": f"{float(input_values[name]):.1f}",
                        "direction": direction,
                        "pct": round(pct, 1),
                    })

        return {
            "score": round(predicted_score, 1),
            "category": category,
            "model_name": metadata.get("model_name", "Unknown"),
            "features": features_list,
        }

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            feature_columns=FEATURE_COLUMNS,
            defaults=defaults,
            prediction=None,
            error=None,
            metadata=metadata,
        )

    @app.post("/predict")
    def predict():
        try:
            raw = {
                "task_completion_hrs": request.form.get("task_completion_hrs", "").strip(),
                "feedback_rating": request.form.get("feedback_rating", "").strip(),
                "attendance_pct": request.form.get("attendance_pct", "").strip(),
            }

            # --- Input validation ---
            errors = []

            # Check for empty fields
            for field, value in raw.items():
                if not value:
                    errors.append(f"{field} is required.")

            # Check for valid numeric values
            parsed = {}
            for field, value in raw.items():
                if not value:
                    continue
                try:
                    parsed[field] = float(value)
                except ValueError:
                    errors.append(f"{field} must be a number, got '{value}'.")

            # Check range bounds
            range_checks = {
                "task_completion_hrs": (2.0, 20.0),
                "feedback_rating": (1.0, 5.0),
                "attendance_pct": (50.0, 100.0),
            }
            for field, (low, high) in range_checks.items():
                if field in parsed:
                    val = parsed[field]
                    if val < low or val > high:
                        errors.append(
                            f"{field} must be between {low} and {high}, got {val}."
                        )

            if errors:
                return render_template(
                    "index.html",
                    feature_columns=FEATURE_COLUMNS,
                    defaults=defaults,
                    prediction=None,
                    error=" | ".join(errors),
                    metadata=metadata,
                ), 400

            payload = {
                "task_completion_hrs": parsed["task_completion_hrs"],
                "feedback_rating": parsed["feedback_rating"],
                "attendance_pct": parsed["attendance_pct"],
            }
            prediction = predict_from_payload(payload)
            return render_template(
                "index.html",
                feature_columns=FEATURE_COLUMNS,
                defaults=defaults,
                prediction=prediction,
                error=None,
                metadata=metadata,
            )
        except Exception as exc:  # pragma: no cover - surfaced to UI
            return render_template(
                "index.html",
                feature_columns=FEATURE_COLUMNS,
                defaults=defaults,
                prediction=None,
                error=str(exc),
                metadata=metadata,
            ), 400

    @app.get("/health")
    def health():
        return {"status": "ok", "model_name": metadata.get("model_name", "Unknown")}

    app.predict_from_payload = predict_from_payload  # type: ignore[attr-defined]
    app.project_paths = paths  # type: ignore[attr-defined]
    return app
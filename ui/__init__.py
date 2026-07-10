"""Flask application factory for the intern performance predictor UI."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, render_template, request

from src.evaluation import classify_performance
from src.model_training import load_model_artifacts
from src.preprocessing import FEATURE_COLUMNS, prepare_prediction_frame


class _ClippedModel:
    """Wrapper that clips regression predictions to the valid score range."""

    def __init__(self, model):
        self._model = model

    def predict(self, X):
        return self._model.predict(X).clip(0, 100)


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


def download_model_from_hf_hub(repo_id: str, token: str | None = None) -> None:
    """Download model artifacts from Hugging Face Hub if not present locally."""
    from huggingface_hub import hf_hub_download

    model_path = Path("models/best_model.pkl")
    metadata_path = Path("models/model_metadata.pkl")
    json_path = Path("models/model_metadata.json")

    if not model_path.exists():
        print("⬇️ Downloading model from Hugging Face Hub...")
        os.makedirs("models", exist_ok=True)
        hf_hub_download(
            repo_id=repo_id,
            filename="models/best_model.pkl",
            repo_type="space",
            token=token,
        )
        print("✅ Model downloaded.")

    if not json_path.exists():
        print("⬇️ Downloading metadata from Hugging Face Hub...")
        hf_hub_download(
            repo_id=repo_id,
            filename="models/model_metadata.json",
            repo_type="space",
            token=token,
        )
        print("✅ Metadata downloaded.")


def ensure_artifacts(paths: dict[str, Path]) -> tuple[object, dict]:
    """Load saved artifacts or download from Hugging Face Hub if none exist.

    On Hugging Face Spaces, the model is downloaded from HF Hub at startup.
    This avoids cold-start training delays and ensures consistent model versions.
    """
    # Try to download from HF Hub if artifacts don't exist
    if not paths["model_path"].exists():
        hf_token = os.environ.get("HF_TOKEN")
        try:
            download_model_from_hf_hub(
                repo_id="awais-dev-ai/Intern-Performance-Predictor",
                token=hf_token if hf_token else None,
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Model artifacts not found at {paths['model_path']}. "
                f"Could not download from Hugging Face Hub: {e}. "
                "Please ensure the model is uploaded to the Space or run 'python main.py'."
            ) from e

    if not paths["metadata_path"].exists():
        hf_token = os.environ.get("HF_TOKEN")
        try:
            download_model_from_hf_hub(
                repo_id="awais-dev-ai/Intern-Performance-Predictor",
                token=hf_token if hf_token else None,
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Metadata not found at {paths['metadata_path']}. "
                f"Could not download from Hugging Face Hub: {e}."
            ) from e

    return load_model_artifacts(
        model_path=paths["model_path"],
        metadata_path=paths["metadata_path"],
    )


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

    raw_model, metadata = ensure_artifacts(paths)
    model = _ClippedModel(raw_model)
    defaults = metadata.get("feature_defaults", {})

    # Use optimized thresholds if available, otherwise use defaults
    threshold_opt = metadata.get("threshold_optimization", {})
    struggle_threshold = threshold_opt.get("struggle_threshold", 40.0)
    excel_threshold = threshold_opt.get("excel_threshold", 75.0)
    metrics = metadata.get("metrics", {}) or {}

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
        if hasattr(raw_model, "feature_importances_"):
            importances = raw_model.feature_importances_
            feature_names = list(FEATURE_COLUMNS)
            input_values = feature_frame.iloc[0]
            baseline = feature_frame.median()

            for i, name in enumerate(feature_names):
                if i < len(importances):
                    deviation = float(input_values[name] - baseline[name])
                    direction = "positive" if deviation >= 0 else "negative"
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
            "score": round(min(max(predicted_score, 0), 100), 1),
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
            metrics=metrics,
            struggle_threshold=struggle_threshold,
            excel_threshold=excel_threshold,
        )

    @app.post("/predict")
    def predict():
        try:
            raw = {
                "task_completion_hrs": request.form.get("task_completion_hrs", "").strip(),
                "feedback_rating": request.form.get("feedback_rating", "").strip(),
                "attendance_pct": request.form.get("attendance_pct", "").strip(),
            }

            errors = []
            for field, value in raw.items():
                if not value:
                    errors.append(f"{field} is required.")

            parsed = {}
            for field, value in raw.items():
                if not value:
                    continue
                try:
                    parsed[field] = float(value)
                except ValueError:
                    errors.append(f"{field} must be a number, got '{value}'.")

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
                    metrics=metrics,
                    struggle_threshold=struggle_threshold,
                    excel_threshold=excel_threshold,
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
                metrics=metrics,
                struggle_threshold=struggle_threshold,
                excel_threshold=excel_threshold,
            )
        except Exception as exc:
            return render_template(
                "index.html",
                feature_columns=FEATURE_COLUMNS,
                defaults=defaults,
                prediction=None,
                error=str(exc),
                metadata=metadata,
                metrics=metrics,
                struggle_threshold=struggle_threshold,
                excel_threshold=excel_threshold,
            ), 400

    @app.get("/health")
    def health():
        return {"status": "ok", "model_name": metadata.get("model_name", "Unknown")}

    app.predict_from_payload = predict_from_payload
    app.project_paths = paths
    return app
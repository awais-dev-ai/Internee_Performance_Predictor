"""Tests for the Flask web application."""

from __future__ import annotations

import ui
from src.data_generation import generate_synthetic_data
from src.model_training import build_model_metadata, save_model_artifacts, select_best_model, train_candidate_models
from src.preprocessing import train_test_split_data


def test_flask_index_and_prediction_flow(tmp_path):
    dataset = generate_synthetic_data(n_samples=160, seed=56)
    split = train_test_split_data(dataset, test_size=0.25, random_state=56)
    fitted_models = train_candidate_models(split.X_train, split.y_train, random_state=56)
    best_result = select_best_model(fitted_models, split.X_test, split.y_test)

    model_path = tmp_path / "models" / "best_model.pkl"
    metadata_path = tmp_path / "models" / "model_metadata.pkl"
    metadata = build_model_metadata(
        model_name=best_result.name,
        metrics=best_result.metrics,
        feature_defaults=split.X_train.median().to_dict(),
    )
    save_model_artifacts(best_result.model, metadata, model_path=model_path, metadata_path=metadata_path)

    app = ui.create_app(base_dir=tmp_path, model_path=model_path, metadata_path=metadata_path)
    client = app.test_client()

    index_response = client.get("/")
    assert index_response.status_code == 200
    assert b"Intern Performance Predictor" in index_response.data

    predict_response = client.post(
        "/predict",
        data={
            "task_completion_hrs": "4.5",
            "feedback_rating": "4.7",
            "attendance_pct": "97.0",
        },
    )

    assert predict_response.status_code == 200
    # Check that the gauge score text and category badge are present
    assert any(label in predict_response.data for label in [b"Excel", b"Average", b"Struggle"])
    # Check that a numeric score appears in the gauge SVG
    import re
    score_match = re.search(rb'data-score="(\d+\.?\d*)"', predict_response.data)
    assert score_match is not None, "Expected a score in the gauge SVG"
    score = float(score_match.group(1))
    assert 0 <= score <= 101  # Allow slight rounding above 100


def test_flask_health_endpoint(tmp_path):
    dataset = generate_synthetic_data(n_samples=80, seed=77)
    split = train_test_split_data(dataset, test_size=0.25, random_state=77)
    fitted_models = train_candidate_models(split.X_train, split.y_train, random_state=77)
    best_result = select_best_model(fitted_models, split.X_test, split.y_test)

    model_path = tmp_path / "models" / "best_model.pkl"
    metadata_path = tmp_path / "models" / "model_metadata.pkl"
    metadata = build_model_metadata(
        model_name=best_result.name,
        metrics=best_result.metrics,
        feature_defaults=split.X_train.median().to_dict(),
    )
    save_model_artifacts(best_result.model, metadata, model_path=model_path, metadata_path=metadata_path)

    app = ui.create_app(base_dir=tmp_path, model_path=model_path, metadata_path=metadata_path)
    client = app.test_client()

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"
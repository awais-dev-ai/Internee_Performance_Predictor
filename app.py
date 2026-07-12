#!/usr/bin/env python
"""Root Flask entry point — re‑exports the app from the ui/ package."""

from __future__ import annotations

import os
import joblib
from huggingface_hub import hf_hub_download
from ui import create_app


def load_model():
    """Download the latest model and metadata from Hugging Face Hub."""
    os.makedirs("models", exist_ok=True)

    model_path = "models/best_model.pkl"
    metadata_pickle_path = "models/model_metadata.pkl"

    # Delete cached files to force a fresh download
    for path in [model_path, metadata_pickle_path]:
        if os.path.exists(path):
            print(f"🗑️ Removing cached {path}...")
            os.remove(path)

    print("⬇️ Downloading latest model from Hugging Face Hub...")
    hf_hub_download(
        repo_id="awais-dev-ai/Intern-Performance-Model",
        filename="best_model.pkl",
        repo_type="model",
        local_dir="models",
    )
    print("✅ Model downloaded.")

    print("⬇️ Downloading latest metadata (pickle) from Hugging Face Hub...")
    try:
        hf_hub_download(
            repo_id="awais-dev-ai/Intern-Performance-Model",
            filename="model_metadata.pkl",
            repo_type="model",
            local_dir="models",
        )
        print("✅ Metadata pickle downloaded.")
    except Exception as e:
        print(f"⚠️ Metadata pickle not found on Hub: {e}")

    # Optional: also download the JSON version (for reference)
    try:
        hf_hub_download(
            repo_id="awais-dev-ai/Intern-Performance-Model",
            filename="model_metadata.json",
            repo_type="model",
            local_dir="models",
        )
        print("✅ Metadata JSON downloaded.")
    except Exception:
        print("⚠️ Metadata JSON not found, using pickle metadata...")

    # ✅ Load the model using joblib (matches how it was saved)
    return joblib.load(model_path)


# Load the model ONCE when the app starts
model = load_model()

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
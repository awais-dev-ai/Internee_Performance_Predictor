#!/usr/bin/env python
"""Root Flask entry point — re-exports the app from the ui/ package."""

from __future__ import annotations

import os
import pickle
from huggingface_hub import hf_hub_download
from ui import create_app

def load_model():
    """Download the latest model from Hugging Face Hub (overwrites local cache)."""
    model_path = "models/best_model.pkl"
    os.makedirs("models", exist_ok=True)
    
    # 🔥 FORCE REDOWNLOAD: Delete the cached file if it exists
    if os.path.exists(model_path):
        print("🗑️ Removing cached model to fetch the latest version...")
        os.remove(model_path)
    
    print("⬇️ Downloading latest model from Hugging Face Hub...")
    hf_hub_download(
        repo_id="awais-dev-ai/Intern-Performance-Predictor",
        filename=model_path,
        local_dir=".",
        token=os.environ.get("HF_TOKEN")  # Optional; only needed for private Spaces
    )
    print("✅ Latest model downloaded.")
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Load the model ONCE when the app starts
model = load_model()

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
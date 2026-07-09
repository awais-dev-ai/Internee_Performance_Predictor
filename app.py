#!/usr/bin/env python
"""Root Flask entry point — re-exports the app from the ui/ package.

Usage:
    python app.py       (development, reads PORT env var)
    flask run
"""

from __future__ import annotations

import os

from ui import create_app

app = create_app()

if __name__ == "__main__":
    # HF Spaces requires port 7860. Fallback to 5000 only for traditional local runs.
    port = int(os.environ.get("PORT", 7860))
    
    # Debug mode defaults to False for safety. Enable via FLASK_DEBUG=true
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

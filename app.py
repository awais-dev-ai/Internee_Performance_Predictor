#!/usr/bin/env python
"""Root Flask entry point — re-exports the app from the ui/ package.

Usage:
    python app.py
    flask run
"""

from __future__ import annotations

from ui import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
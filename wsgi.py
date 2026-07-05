"""WSGI entrypoint for production servers (Gunicorn, Waitress, etc.).

Usage:
    waitress-serve wsgi:app
    gunicorn wsgi:app
"""

from __future__ import annotations

from ui import create_app

app = create_app()
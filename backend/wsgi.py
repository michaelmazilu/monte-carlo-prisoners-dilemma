"""WSGI entrypoint for production servers such as gunicorn."""

from .app import create_app

app = create_app()

"""
Backend package exposing the Flask application factory for the
Monte Carlo Prisoner's Dilemma simulator MVP.
"""

from .app import create_app

__all__ = ["create_app"]

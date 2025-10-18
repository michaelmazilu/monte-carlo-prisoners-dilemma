"""
Flask application exposing REST endpoints and an SSE stream for the
Monte Carlo Prisoner's Dilemma MVP.
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Dict, Iterable, Tuple

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

from .simulation import (
    SimulationConfig,
    SimulationValidationError,
    StrategyConfig,
    StrategyType,
    run_simulation,
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"

_SESSION_STORE: Dict[str, SimulationConfig] = {}
_SESSION_LOCK = threading.Lock()


def create_app() -> Flask:
    """Application factory used by both development and production runners."""
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
    app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

    @app.get("/")
    def index() -> Response:
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.get("/health")
    def health() -> Response:
        return jsonify({"status": "ok"})

    @app.get("/api/strategies")
    def list_strategies() -> Response:
        data = [
            {
                "id": StrategyType.ALWAYS_COOPERATE.value,
                "label": "Always Cooperate",
                "requires_probability": False,
            },
            {
                "id": StrategyType.ALWAYS_DEFECT.value,
                "label": "Always Defect",
                "requires_probability": False,
            },
            {
                "id": StrategyType.PROBABILISTIC.value,
                "label": "Probabilistic",
                "requires_probability": True,
            },
        ]
        return jsonify({"strategies": data})

    @app.post("/api/simulations")
    def create_simulation() -> Response:
        payload = request.get_json(silent=True) or {}
        try:
            config = _parse_simulation_config(payload)
        except SimulationValidationError as exc:
            return jsonify({"error": str(exc)}), 400

        simulation_id = str(uuid.uuid4())
        with _SESSION_LOCK:
            _SESSION_STORE[simulation_id] = config
        return jsonify({"simulation_id": simulation_id})

    @app.get("/api/simulations/<simulation_id>/stream")
    def stream_simulation(simulation_id: str) -> Response:
        with _SESSION_LOCK:
            config = _SESSION_STORE.pop(simulation_id, None)
        if config is None:
            return jsonify({"error": "Unknown simulation id"}), 404

        def event_stream() -> Iterable[str]:
            for event_name, payload in run_simulation(config):
                yield _format_sse(event_name, payload)

        response = Response(stream_with_context(event_stream()), mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
        response.headers["Connection"] = "keep-alive"
        return response

    return app


def _parse_simulation_config(payload: Dict[str, object]) -> SimulationConfig:
    try:
        rounds = int(payload.get("rounds", 0))
        monte_carlo_runs = int(payload.get("monte_carlo_runs", 1))
        raw_strategies = payload.get("strategies") or []
    except (TypeError, ValueError) as exc:
        raise SimulationValidationError("Invalid numeric parameters.") from exc

    if len(raw_strategies) != 2:
        raise SimulationValidationError("Two player strategies are required.")

    strategies = tuple(_parse_strategy_config(raw, index + 1) for index, raw in enumerate(raw_strategies))
    return SimulationConfig(rounds=rounds, monte_carlo_runs=monte_carlo_runs, player_strategies=strategies)  # type: ignore[arg-type]


def _parse_strategy_config(raw: Dict[str, object], player_index: int) -> StrategyConfig:
    try:
        strategy_key = str(raw.get("type", "")).lower()
    except Exception as exc:  # pragma: no cover - defensive
        raise SimulationValidationError(f"Invalid strategy for player {player_index}.") from exc

    if strategy_key not in StrategyType._value2member_map_:
        raise SimulationValidationError(f"Unsupported strategy '{strategy_key}' for player {player_index}.")

    strategy_type = StrategyType(strategy_key)
    probability = 1.0

    if strategy_type is StrategyType.PROBABILISTIC:
        if "cooperate_probability" not in raw:
            raise SimulationValidationError(
                f"Missing cooperate_probability for probabilistic strategy (player {player_index})."
            )
        probability_value = raw["cooperate_probability"]
        try:
            probability = float(probability_value)
        except (TypeError, ValueError) as exc:
            raise SimulationValidationError(
                f"Invalid cooperate_probability for player {player_index}."
            ) from exc
        if probability > 1.0:
            # Allow frontend to send percentages (0-100) and convert them.
            probability /= 100.0

    return StrategyConfig(strategy_type=strategy_type, cooperate_probability=probability)


def _format_sse(event: str, payload: Dict[str, object]) -> str:
    """
    Format a payload as an SSE event block.

    Each block ends with a blank line per the SSE specification.
    """
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

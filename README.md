# Monte Carlo Prisoner's Dilemma Simulator (MVP)

A full-stack MVP for exploring the Prisoner's Dilemma with live visualisations. Configure each player's strategy, run single-shot or Monte Carlo simulations, and watch real-time charts update every round via Server-Sent Events (SSE).

## Features
- **Configurable strategies per player:** always cooperate, always defect, probabilistic with custom cooperation rates, Tit for Tat, or Random.
- **Simulation controls:** choose rounds and optional Monte Carlo runs for aggregated statistics.
- **Live analytics:** interactive Chart.js visualisations for cumulative payoff, per-round payoff, cooperation rate, and outcome distribution.
- **Backend streaming:** Flask 3 + PyTorch 2.8 pipeline emitting SSE updates after every round.
- **Container-ready:** Docker + docker-compose + Nginx reverse proxy for deployment.

## Project Structure
```
backend/          # Flask app, SSE endpoint, simulation engine
frontend/         # Static HTML/CSS/JS powered by Chart.js
nginx/            # Reverse proxy configuration
requirements.txt  # Python dependencies (Python 3.13+)
Dockerfile        # Backend container image
```

## Getting Started (Local Python)
1. **Create a virtual environment** (Python 3.13 suggested):
   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   flask --app backend.app:create_app run --host 0.0.0.0 --port 8000 --debug
   ```
4. Open http://localhost:8000 in your browser — Flask serves both the API and the frontend assets in development.

## Running with Docker
```bash
docker compose up --build
```
- Backend is available at http://localhost:8000.
- Nginx serves the frontend at http://localhost:8080 and proxies `/api/*` SSE requests to the backend.
- Gunicorn uses the threaded worker class by default; no gevent dependency is required for Python 3.13 compatibility.

## Testing
Run the unit test suite (Python unittest):
```bash
python -m unittest discover -s tests -p "test_*.py"
```

## API Overview
- `POST /api/simulations` — start a simulation, returns `{ "simulation_id": "..." }`.
- `GET /api/simulations/<id>/stream` — open SSE stream emitting `round`, `run_complete`, and `summary` events.
- `GET /api/strategies` — list supported strategies for UI clients.
- `GET /health` — health probe for monitoring.

Each `round` event payload contains actions, payoffs, cumulative totals, cooperation rates, and outcome counts. The `summary` event delivers final totals and averaged statistics across all Monte Carlo runs.

## Notes
- PyTorch powers the simulation to keep the computation vectorised and ready for GPU acceleration if desired.
- SSE streaming requires servers (like Nginx) to disable buffering. The provided configuration already handles this.
- The probabilistic strategy accepts either a decimal probability in `[0, 1]` or a percentage in `[0, 100]` from the client.
- If you previously installed gevent, it is no longer required; the project now runs with Gunicorn's threaded worker class which works out of the box on Python 3.13.

Enjoy exploring strategic choices in the Prisoner's Dilemma!

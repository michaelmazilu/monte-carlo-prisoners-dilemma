from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import torch
import numpy as np
import json
import time
import threading
import queue

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Globals for SSE streaming single-run
run_queue = queue.Queue()
run_thread = None
current_run = None

@app.route("/")
def home():
    return """
    <h1>🎯 Monte Carlo Prisoner's Dilemma Simulator</h1>
    <p>Welcome to the Prisoner's Dilemma simulation API!</p>
    <h2>Available Endpoints:</h2>
    <ul>
        <li><strong>POST /simulate</strong> - Run a single Monte Carlo simulation</li>
        <li><strong>POST /parameter_sweep</strong> - Run 10,000 experiments (100x100 grid)</li>
        <li><strong>POST /batch_simulate</strong> - Run multiple custom simulations</li>
        <li><strong>GET /strategies</strong> - Get available strategy types</li>
    </ul>
    <h2>Parameter Sweep Experiment:</h2>
    <p>Run comprehensive analysis with:</p>
    <ul>
        <li>Player 1: 100% → 0% cooperation (1% steps)</li>
        <li>Player 2: 0% → 100% cooperation (1% steps)</li>
        <li>100 rounds per configuration</li>
        <li>Total: 10,000 experiments</li>
    </ul>
    <h2>How to use:</h2>
    <p>Send a POST request to <code>/parameter_sweep</code> with JSON data:</p>
    <pre>
{
    "rounds_per_config": 100,
    "step_size": 0.01
}
    </pre>
    """

@app.route("/start_run", methods=["POST"])
def start_run():
    """Start a PyTorch simulation in background and stream progress via /simulate_stream."""
    global run_thread, current_run

    if current_run and current_run.get('status') == 'running':
        return jsonify({"error": "A simulation is already running"}), 400

    data = request.json or {}
    p1 = float(data.get("player1_prob", 0.5))  # cooperation probability
    p2 = float(data.get("player2_prob", 0.5))
    rounds = int(data.get("rounds", 1000))
    batch_size = int(data.get("batch_size", max(10, rounds // 100)))

    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        return jsonify({"error": "Probabilities must be between 0 and 1"}), 400
    if rounds < 0 or rounds > 50000:
        return jsonify({"error": "Rounds must be between 0 and 50,000"}), 400

    # Clear stale messages from previous runs
    try:
        while True:
            run_queue.get_nowait()
    except queue.Empty:
        pass

    def progress_callback(payload):
        # Ensure JSON-serializable types
        run_queue.put(payload)

    current_run = {"status": "running", "rounds": rounds, "completed": 0, "start_time": time.time()}
    run_thread = threading.Thread(target=run_simulation_streaming, args=(p1, p2, rounds, batch_size, progress_callback))
    run_thread.daemon = True
    run_thread.start()

    return jsonify({"message": "Simulation started"})

@app.route("/simulate_stream")
def simulate_stream():
    """SSE stream for live simulation updates. If no run is active, start one from query params."""
    global run_thread, current_run

    if not (current_run and current_run.get('status') == 'running'):
        # optional auto-start using query params
        try:
            while True:
                run_queue.get_nowait()
        except queue.Empty:
            pass

        p1 = float(request.args.get('p1', 0.5))
        p2 = float(request.args.get('p2', 0.5))
        rounds = int(request.args.get('rounds', 1000))
        batch_size = int(request.args.get('batch', max(10, rounds // 100)))

        def progress_callback(payload):
            run_queue.put(payload)

        current_run = {"status": "running", "rounds": rounds, "completed": 0, "start_time": time.time()}
        run_thread = threading.Thread(target=run_simulation_streaming, args=(p1, p2, rounds, batch_size, progress_callback))
        run_thread.daemon = True
        run_thread.start()

    def generate():
        while True:
            try:
                payload = run_queue.get(timeout=1)
                yield f"data: {json.dumps(payload)}\n\n"
                if payload.get('type') in ('complete', 'error'):
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
                break
    return Response(generate(), mimetype='text/event-stream')

def run_monte_carlo_simulation(p1, p2, rounds, strategy1="probabilistic", strategy2="probabilistic"):
    """Run Monte Carlo simulation for Prisoner's Dilemma"""
    
    # Generate actions based on strategies
    if strategy1 == "probabilistic":
        p1_actions = torch.rand(rounds) < p1
    elif strategy1 == "always_cooperate":
        p1_actions = torch.ones(rounds, dtype=torch.bool)
    elif strategy1 == "always_defect":
        p1_actions = torch.zeros(rounds, dtype=torch.bool)
    else:
        p1_actions = torch.rand(rounds) < p1  # Default to probabilistic
    
    if strategy2 == "probabilistic":
        p2_actions = torch.rand(rounds) < p2
    elif strategy2 == "always_cooperate":
        p2_actions = torch.ones(rounds, dtype=torch.bool)
    elif strategy2 == "always_defect":
        p2_actions = torch.zeros(rounds, dtype=torch.bool)
    else:
        p2_actions = torch.rand(rounds) < p2  # Default to probabilistic

    # Payoff matrix: (C=1, D=0)
    # Both cooperate: (3, 3)
    # P1 cooperates, P2 defects: (0, 5)
    # P1 defects, P2 cooperates: (5, 0)
    # Both defect: (1, 1)
    payoffs = torch.zeros((rounds, 2))
    # Vectorized float-mask payoff computation (avoids torch.where scalar dtype issues)
    p1_pay = (
        (p1_actions & p2_actions).float() * 3.0 +
        (p1_actions & (~p2_actions)).float() * 0.0 +
        ((~p1_actions) & p2_actions).float() * 5.0 +
        ((~p1_actions) & (~p2_actions)).float() * 1.0
    )
    p2_pay = (
        (p1_actions & p2_actions).float() * 3.0 +
        (p1_actions & (~p2_actions)).float() * 5.0 +
        ((~p1_actions) & p2_actions).float() * 0.0 +
        ((~p1_actions) & (~p2_actions)).float() * 1.0
    )
    payoffs[:, 0] = p1_pay
    payoffs[:, 1] = p2_pay

    # Calculate statistics
    p1_coop_rate = float(p1_actions.sum()) / rounds
    p2_coop_rate = float(p2_actions.sum()) / rounds
    
    # Calculate cooperation outcomes
    both_cooperate = float((p1_actions & p2_actions).sum()) / rounds
    both_defect = float((~p1_actions & ~p2_actions).sum()) / rounds
    p1_coop_p2_defect = float((p1_actions & ~p2_actions).sum()) / rounds
    p1_defect_p2_coop = float((~p1_actions & p2_actions).sum()) / rounds

    result = {
        "player1_avg_payoff": payoffs[:, 0].float().mean().item(),
        "player2_avg_payoff": payoffs[:, 1].float().mean().item(),
        "player1_coop_rate": p1_coop_rate,
        "player2_coop_rate": p2_coop_rate,
        "both_cooperate_rate": both_cooperate,
        "both_defect_rate": both_defect,
        "p1_coop_p2_defect_rate": p1_coop_p2_defect,
        "p1_defect_p2_coop_rate": p1_defect_p2_coop,
        "total_rounds": rounds,
        "strategy1": strategy1,
        "strategy2": strategy2,
        "parameters": {
            "player1_prob": p1,
            "player2_prob": p2
        }
    }
    
    return result


def run_simulation_streaming(p1, p2, rounds, batch_size, progress_callback):
    """Run batches in PyTorch and stream progress via callback."""
    global current_run
    try:
        current_run = {"status": "running", "rounds": rounds, "completed": 0, "start_time": time.time()}

        if rounds == 0:
            summary = run_monte_carlo_simulation(p1, p2, 0)
            progress_callback({"type": "complete", "result": summary})
            current_run["status"] = "completed"
            return

        p1_total = 0.0
        p2_total = 0.0
        both_coop = 0
        both_defect = 0
        p1_coop_p2_defect = 0
        p1_defect_p2_coop = 0

        remaining = rounds
        completed = 0
        while remaining > 0 and current_run.get('status') == 'running':
            b = min(batch_size, remaining)

            # Sample cooperation according to probabilities
            p1_actions = torch.rand(b) < p1
            p2_actions = torch.rand(b) < p2

            # Outcome counts
            bc = (p1_actions & p2_actions).sum().item()
            bd = ((~p1_actions) & (~p2_actions)).sum().item()
            c_d = (p1_actions & (~p2_actions)).sum().item()
            d_c = ((~p1_actions) & p2_actions).sum().item()
            both_coop += int(bc)
            both_defect += int(bd)
            p1_coop_p2_defect += int(c_d)
            p1_defect_p2_coop += int(d_c)

            # Payoffs
            p1_pay = (
                (p1_actions & p2_actions).float() * 3.0 +
                (p1_actions & (~p2_actions)).float() * 0.0 +
                ((~p1_actions) & p2_actions).float() * 5.0 +
                ((~p1_actions) & (~p2_actions)).float() * 1.0
            )
            p2_pay = (
                (p1_actions & p2_actions).float() * 3.0 +
                (p1_actions & (~p2_actions)).float() * 5.0 +
                ((~p1_actions) & p2_actions).float() * 0.0 +
                ((~p1_actions) & (~p2_actions)).float() * 1.0
            )
            p1_total += float(p1_pay.sum().item())
            p2_total += float(p2_pay.sum().item())

            completed += b
            remaining -= b
            progress = (completed / rounds) * 100.0
            metrics = {
                "p1_avg": p1_total / completed,
                "p2_avg": p2_total / completed,
                "total_avg": (p1_total + p2_total) / completed,
                "both_cooperate": both_coop,
                "both_defect": both_defect,
                "p1_coop_p2_defect": p1_coop_p2_defect,
                "p1_defect_p2_coop": p1_defect_p2_coop
            }
            progress_callback({
                "type": "progress",
                "completed": completed,
                "total": rounds,
                "progress": progress,
                "metrics": metrics
            })

        if current_run.get('status') == 'running':
            summary = {
                "player1_avg_payoff": p1_total / rounds if rounds > 0 else 0.0,
                "player2_avg_payoff": p2_total / rounds if rounds > 0 else 0.0,
                "player1_coop_rate": (both_coop + p1_coop_p2_defect) / rounds if rounds > 0 else 0.0,
                "player2_coop_rate": (both_coop + p1_defect_p2_coop) / rounds if rounds > 0 else 0.0,
                "both_cooperate_rate": both_coop / rounds if rounds > 0 else 0.0,
                "both_defect_rate": both_defect / rounds if rounds > 0 else 0.0,
                "p1_coop_p2_defect_rate": p1_coop_p2_defect / rounds if rounds > 0 else 0.0,
                "p1_defect_p2_coop_rate": p1_defect_p2_coop / rounds if rounds > 0 else 0.0,
                "total_rounds": rounds,
                "parameters": {"player1_prob": p1, "player2_prob": p2}
            }
            progress_callback({"type": "complete", "result": summary})
            current_run["status"] = "completed"
    except Exception as e:
        if current_run is not None:
            current_run["status"] = "error"
        progress_callback({"type": "error", "error": str(e)})

@app.route("/strategies", methods=["GET"])
def get_strategies():
    """Get available strategies"""
    strategies = {
        "probabilistic": "Random cooperation based on probability",
        "always_cooperate": "Always cooperate (Tit-for-Tat)",
        "always_defect": "Always defect (Always betray)"
    }
    return jsonify(strategies)

@app.route("/parameter_sweep", methods=["POST"])
def parameter_sweep():
    """Run comprehensive parameter sweep: 100x100 grid of cooperation probabilities"""
    try:
        data = request.json
        rounds_per_config = data.get("rounds_per_config", 100)
        step_size = data.get("step_size", 0.01)  # 1% steps
        
        # Player 1: 100% → 0% cooperation (decreasing)
        # Player 2: 0% → 100% cooperation (increasing)
        p1_probs = torch.linspace(1.0, 0.0, int(1.0/step_size) + 1)
        p2_probs = torch.linspace(0.0, 1.0, int(1.0/step_size) + 1)
        
        results = []
        total_configs = len(p1_probs) * len(p2_probs)
        
        print(f"Running {total_configs} configurations with {rounds_per_config} rounds each...")
        
        for i, p1 in enumerate(p1_probs):
            for j, p2 in enumerate(p2_probs):
                # Run simulation
                result = run_monte_carlo_simulation(
                    p1.item(), p2.item(), rounds_per_config, 
                    "probabilistic", "probabilistic"
                )
                
                # Add configuration info
                result.update({
                    "p1_prob": p1.item(),
                    "p2_prob": p2.item(),
                    "config_id": i * len(p2_probs) + j,
                    "total_configs": total_configs
                })
                
                results.append(result)
                
                # Progress indicator
                if (i * len(p2_probs) + j) % 1000 == 0:
                    print(f"Completed {i * len(p2_probs) + j}/{total_configs} configurations")
        
        print(f"Parameter sweep complete! Processed {total_configs} configurations.")
        
        return jsonify({
            "results": results,
            "summary": {
                "total_configurations": total_configs,
                "rounds_per_config": rounds_per_config,
                "step_size": step_size,
                "p1_range": [p1_probs[0].item(), p1_probs[-1].item()],
                "p2_range": [p2_probs[0].item(), p2_probs[-1].item()]
            }
        })
    
    except Exception as e:
        return jsonify({"error": f"Parameter sweep failed: {str(e)}"}), 500

@app.route("/batch_simulate", methods=["POST"])
def batch_simulate():
    """Run multiple simulations with different parameters"""
    try:
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
        
        data = request.json
        simulations = data.get("simulations", [])
        
        if not simulations:
            return jsonify({"error": "No simulations provided"}), 400
        
        results = []
        for sim in simulations:
            try:
                result = run_monte_carlo_simulation(
                    sim.get("player1_prob", 0.5),
                    sim.get("player2_prob", 0.5),
                    sim.get("rounds", 1000),
                    sim.get("strategy1", "probabilistic"),
                    sim.get("strategy2", "probabilistic")
                )
                results.append(result)
            except Exception as e:
                results.append({"error": f"Simulation failed: {str(e)}"})
        
        return jsonify({"results": results})
    
    except Exception as e:
        return jsonify({"error": f"Batch simulation failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)

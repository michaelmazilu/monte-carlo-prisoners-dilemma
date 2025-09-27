from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

@app.route("/")
def home():
    return """
    <h1>Monte Carlo Prisoner's Dilemma Simulator</h1>
    <p>Welcome to the Prisoner's Dilemma simulation API!</p>
    <h2>Available Endpoints:</h2>
    <ul>
        <li><strong>POST /simulate</strong> - Run a Monte Carlo simulation</li>
    </ul>
    <h2>How to use:</h2>
    <p>Send a POST request to <code>/simulate</code> with JSON data:</p>
    <pre>
{
    "player1_prob": 0.5,
    "player2_prob": 0.5,
    "rounds": 1000
}
    </pre>
    """

@app.route("/simulate", methods=["POST"])
def simulate():
    try:
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
        
        data = request.json
        p1 = data.get("player1_prob", 0.5)
        p2 = data.get("player2_prob", 0.5)
        rounds = data.get("rounds", 1000)
        strategy1 = data.get("strategy1", "probabilistic")
        strategy2 = data.get("strategy2", "probabilistic")

        # Input validation
        if not (0 <= p1 <= 1) or not (0 <= p2 <= 1):
            return jsonify({"error": "Probabilities must be between 0 and 1"}), 400
        
        if rounds <= 0 or rounds > 1000000:
            return jsonify({"error": "Rounds must be between 1 and 1,000,000"}), 400

        # Run Monte Carlo simulation
        result = run_monte_carlo_simulation(p1, p2, rounds, strategy1, strategy2)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"Simulation failed: {str(e)}"}), 500

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
    
    # Player 1 payoffs
    payoffs[:, 0] = torch.where(
        p1_actions & p2_actions, 3,  # Both cooperate
        torch.where(p1_actions & ~p2_actions, 0,  # P1 cooperates, P2 defects
                   torch.where(~p1_actions & p2_actions, 5, 1))  # P1 defects, P2 cooperates; Both defect
    )
    
    # Player 2 payoffs
    payoffs[:, 1] = torch.where(
        p1_actions & p2_actions, 3,  # Both cooperate
        torch.where(p1_actions & ~p2_actions, 5,  # P1 cooperates, P2 defects
                   torch.where(~p1_actions & p2_actions, 0, 1))  # P1 defects, P2 cooperates; Both defect
    )

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

@app.route("/strategies", methods=["GET"])
def get_strategies():
    """Get available strategies"""
    strategies = {
        "probabilistic": "Random cooperation based on probability",
        "always_cooperate": "Always cooperate (Tit-for-Tat)",
        "always_defect": "Always defect (Always betray)"
    }
    return jsonify(strategies)

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

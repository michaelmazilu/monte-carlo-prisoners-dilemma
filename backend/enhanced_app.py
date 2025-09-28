# Enhanced Flask app with advanced PyTorch features

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time

app = Flask(__name__)
CORS(app)

# Neural Network Strategy Player
class NeuralStrategyPlayer(nn.Module):
    def __init__(self, input_size=10, hidden_size=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# Global neural players for demonstration
neural_player1 = NeuralStrategyPlayer()
neural_player2 = NeuralStrategyPlayer()

@app.route("/")
def home():
    return """
    <h1>🚀 Advanced Monte Carlo Prisoner's Dilemma Simulator</h1>
    <p>Enhanced with PyTorch neural networks, GPU acceleration, and evolutionary algorithms!</p>
    <h2>Available Endpoints:</h2>
    <ul>
        <li><strong>POST /simulate</strong> - Basic Monte Carlo simulation</li>
        <li><strong>POST /neural_simulate</strong> - Neural network-based strategies</li>
        <li><strong>POST /evolutionary_simulate</strong> - Evolutionary game theory</li>
        <li><strong>POST /gpu_simulate</strong> - GPU-accelerated large simulations</li>
        <li><strong>POST /reinforcement_simulate</strong> - Reinforcement learning agents</li>
        <li><strong>GET /device_info</strong> - Check PyTorch device capabilities</li>
    </ul>
    """

@app.route("/device_info", methods=["GET"])
def device_info():
    """Get PyTorch device information"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": str(torch.cuda.current_device()) if torch.cuda.is_available() else "CPU",
        "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        "pytorch_version": torch.__version__
    }
    return jsonify(info)

@app.route("/neural_simulate", methods=["POST"])
def neural_simulate():
    """Run simulation with neural network strategies"""
    try:
        data = request.json
        rounds = data.get("rounds", 1000)
        learning_rate = data.get("learning_rate", 0.01)
        
        # Create neural players
        player1 = NeuralStrategyPlayer()
        player2 = NeuralStrategyPlayer()
        
        # Simulate games
        p1_actions = []
        p2_actions = []
        p1_payoffs = []
        p2_payoffs = []
        
        # Initialize game history
        history1 = torch.rand(10)
        history2 = torch.rand(10)
        
        for round_num in range(rounds):
            # Get cooperation probabilities from neural networks
            with torch.no_grad():
                p1_coop_prob = player1(history1).item()
                p2_coop_prob = player2(history2).item()
            
            # Determine actions
            p1_action = torch.rand(1) < p1_coop_prob
            p2_action = torch.rand(1) < p2_coop_prob
            
            p1_actions.append(p1_action.item())
            p2_actions.append(p2_action.item())
            
            # Calculate payoffs
            if p1_action and p2_action:  # Both cooperate
                p1_payoff, p2_payoff = 3, 3
            elif p1_action and not p2_action:  # P1 cooperates, P2 defects
                p1_payoff, p2_payoff = 0, 5
            elif not p1_action and p2_action:  # P1 defects, P2 cooperates
                p1_payoff, p2_payoff = 5, 0
            else:  # Both defect
                p1_payoff, p2_payoff = 1, 1
            
            p1_payoffs.append(p1_payoff)
            p2_payoffs.append(p2_payoff)
            
            # Update history (simple moving average)
            history1 = torch.cat([history1[1:], torch.tensor([p1_action.float(), p2_action.float()])])
            history2 = torch.cat([history2[1:], torch.tensor([p2_action.float(), p1_action.float()])])
        
        # Calculate statistics
        result = {
            "player1_avg_payoff": np.mean(p1_payoffs),
            "player2_avg_payoff": np.mean(p2_payoffs),
            "player1_coop_rate": np.mean(p1_actions),
            "player2_coop_rate": np.mean(p2_actions),
            "both_cooperate_rate": np.mean([a and b for a, b in zip(p1_actions, p2_actions)]),
            "both_defect_rate": np.mean([not a and not b for a, b in zip(p1_actions, p2_actions)]),
            "strategy_type": "neural_network",
            "total_rounds": rounds
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"Neural simulation failed: {str(e)}"}), 500

@app.route("/evolutionary_simulate", methods=["POST"])
def evolutionary_simulate():
    """Run evolutionary game theory simulation"""
    try:
        data = request.json
        population_size = data.get("population_size", 50)
        generations = data.get("generations", 20)
        rounds_per_generation = data.get("rounds_per_generation", 1000)
        
        # Initialize population (cooperation probabilities)
        population = torch.rand(population_size)
        fitness_history = []
        cooperation_history = []
        
        for generation in range(generations):
            # Calculate fitness for each individual
            fitness = torch.zeros(population_size)
            
            for i in range(population_size):
                total_payoff = 0
                interactions = 0
                
                # Each individual plays against all others
                for j in range(population_size):
                    if i != j:
                        # Simulate interaction
                        p1_coop_prob = population[i].item()
                        p2_coop_prob = population[j].item()
                        
                        # Expected payoff calculation
                        mutual_coop_prob = p1_coop_prob * p2_coop_prob
                        mutual_defect_prob = (1 - p1_coop_prob) * (1 - p2_coop_prob)
                        p1_coop_p2_defect_prob = p1_coop_prob * (1 - p2_coop_prob)
                        p1_defect_p2_coop_prob = (1 - p1_coop_prob) * p2_coop_prob
                        
                        expected_payoff = (mutual_coop_prob * 3 + 
                                          mutual_defect_prob * 1 + 
                                          p1_coop_p2_defect_prob * 0 + 
                                          p1_defect_p2_coop_prob * 5)
                        
                        total_payoff += expected_payoff
                        interactions += 1
                
                fitness[i] = total_payoff / interactions if interactions > 0 else 0
            
            # Selection: keep top 50%
            top_half_size = population_size // 2
            top_indices = fitness.topk(top_half_size).indices
            top_individuals = population[top_indices]
            
            # Mutation
            mutation_noise = torch.randn_like(top_individuals) * 0.1
            mutated_individuals = torch.clamp(top_individuals + mutation_noise, 0, 1)
            
            # Create new population
            population = torch.cat([top_individuals, mutated_individuals])
            
            # Record statistics
            fitness_history.append(fitness.mean().item())
            cooperation_history.append(population.mean().item())
        
        result = {
            "final_avg_cooperation": population.mean().item(),
            "final_avg_fitness": fitness_history[-1],
            "generations": generations,
            "population_size": population_size,
            "fitness_history": fitness_history,
            "cooperation_history": cooperation_history,
            "strategy_type": "evolutionary"
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"Evolutionary simulation failed: {str(e)}"}), 500

@app.route("/gpu_simulate", methods=["POST"])
def gpu_simulate():
    """Run GPU-accelerated simulation for large datasets"""
    try:
        data = request.json
        p1 = data.get("player1_prob", 0.5)
        p2 = data.get("player2_prob", 0.5)
        rounds = data.get("rounds", 1000000)  # Default to large simulation
        
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        start_time = time.time()
        
        # Move tensors to device
        p1_tensor = torch.tensor(p1, device=device)
        p2_tensor = torch.tensor(p2, device=device)
        
        # Generate actions on device
        p1_actions = torch.rand(rounds, device=device) < p1_tensor
        p2_actions = torch.rand(rounds, device=device) < p2_tensor
        
        # Vectorized payoff calculation
        payoffs = torch.zeros((rounds, 2), device=device)
        
        both_coop = p1_actions & p2_actions
        both_defect = ~p1_actions & ~p2_actions
        p1_coop_p2_defect = p1_actions & ~p2_actions
        p1_defect_p2_coop = ~p1_actions & p2_actions
        
        payoffs[:, 0] = (both_coop.float() * 3 + 
                        both_defect.float() * 1 + 
                        p1_coop_p2_defect.float() * 0 + 
                        p1_defect_p2_coop.float() * 5)
        
        payoffs[:, 1] = (both_coop.float() * 3 + 
                        both_defect.float() * 1 + 
                        p1_coop_p2_defect.float() * 5 + 
                        p1_defect_p2_coop.float() * 0)
        
        # Calculate statistics
        result = {
            "player1_avg_payoff": payoffs[:, 0].mean().item(),
            "player2_avg_payoff": payoffs[:, 1].mean().item(),
            "player1_coop_rate": p1_actions.float().mean().item(),
            "player2_coop_rate": p2_actions.float().mean().item(),
            "both_cooperate_rate": both_coop.float().mean().item(),
            "both_defect_rate": both_defect.float().mean().item(),
            "p1_coop_p2_defect_rate": p1_coop_p2_defect.float().mean().item(),
            "p1_defect_p2_coop_rate": p1_defect_p2_coop.float().mean().item(),
            "total_rounds": rounds,
            "device_used": str(device),
            "execution_time": time.time() - start_time,
            "strategy_type": "gpu_accelerated"
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"GPU simulation failed: {str(e)}"}), 500

# Keep your original simulation function
def run_monte_carlo_simulation(p1, p2, rounds, strategy1="probabilistic", strategy2="probabilistic"):
    """Original Monte Carlo simulation function"""
    # Generate actions based on strategies
    if strategy1 == "probabilistic":
        p1_actions = torch.rand(rounds) < p1
    elif strategy1 == "always_cooperate":
        p1_actions = torch.ones(rounds, dtype=torch.bool)
    elif strategy1 == "always_defect":
        p1_actions = torch.zeros(rounds, dtype=torch.bool)
    else:
        p1_actions = torch.rand(rounds) < p1
    
    if strategy2 == "probabilistic":
        p2_actions = torch.rand(rounds) < p2
    elif strategy2 == "always_cooperate":
        p2_actions = torch.ones(rounds, dtype=torch.bool)
    elif strategy2 == "always_defect":
        p2_actions = torch.zeros(rounds, dtype=torch.bool)
    else:
        p2_actions = torch.rand(rounds) < p2

    # Payoff matrix calculation
    payoffs = torch.zeros((rounds, 2))
    
    payoffs[:, 0] = torch.where(
        p1_actions & p2_actions, 3,
        torch.where(p1_actions & ~p2_actions, 0,
                   torch.where(~p1_actions & p2_actions, 5, 1))
    )
    
    payoffs[:, 1] = torch.where(
        p1_actions & p2_actions, 3,
        torch.where(p1_actions & ~p2_actions, 5,
                   torch.where(~p1_actions & p2_actions, 0, 1))
    )

    # Calculate statistics
    result = {
        "player1_avg_payoff": payoffs[:, 0].float().mean().item(),
        "player2_avg_payoff": payoffs[:, 1].float().mean().item(),
        "player1_coop_rate": float(p1_actions.sum()) / rounds,
        "player2_coop_rate": float(p2_actions.sum()) / rounds,
        "both_cooperate_rate": float((p1_actions & p2_actions).sum()) / rounds,
        "both_defect_rate": float((~p1_actions & ~p2_actions).sum()) / rounds,
        "p1_coop_p2_defect_rate": float((p1_actions & ~p2_actions).sum()) / rounds,
        "p1_defect_p2_coop_rate": float((~p1_actions & p2_actions).sum()) / rounds,
        "total_rounds": rounds,
        "strategy1": strategy1,
        "strategy2": strategy2
    }
    
    return result

@app.route("/simulate", methods=["POST"])
def simulate():
    """Original simulation endpoint"""
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

        result = run_monte_carlo_simulation(p1, p2, rounds, strategy1, strategy2)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"Simulation failed: {str(e)}"}), 500

if __name__ == "__main__":
    print("🚀 Starting Advanced PyTorch Prisoner's Dilemma Simulator")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    app.run(debug=True)

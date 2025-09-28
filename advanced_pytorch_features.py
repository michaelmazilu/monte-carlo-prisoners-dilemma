# Advanced PyTorch Features for Prisoner's Dilemma Simulator

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple

class NeuralStrategyPlayer(nn.Module):
    """Neural network-based strategy player that learns optimal cooperation probabilities"""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Output cooperation probability
        )
    
    def forward(self, x):
        return self.network(x)
    
    def get_cooperation_prob(self, game_history: torch.Tensor) -> float:
        """Get cooperation probability based on game history"""
        with torch.no_grad():
            prob = self.forward(game_history)
            return prob.item()

class EvolutionarySimulator:
    """Use PyTorch for evolutionary game theory simulations"""
    
    def __init__(self, population_size: int = 100):
        self.population_size = population_size
        self.population = torch.rand(population_size)  # Cooperation probabilities
        self.fitness = torch.zeros(population_size)
    
    def evolve_generation(self, rounds: int = 1000) -> Dict:
        """Evolve one generation using PyTorch operations"""
        # Create payoff matrix for all pairwise interactions
        payoffs = self._calculate_pairwise_payoffs(rounds)
        
        # Calculate fitness (average payoff)
        self.fitness = payoffs.mean(dim=1)
        
        # Selection: keep top 50% and mutate
        top_half = self.fitness.topk(self.population_size // 2).indices
        new_population = self.population[top_half].clone()
        
        # Mutation with PyTorch
        mutation_noise = torch.randn_like(new_population) * 0.1
        new_population = torch.clamp(new_population + mutation_noise, 0, 1)
        
        # Replace bottom half with mutated top half
        self.population = torch.cat([new_population, new_population])
        
        return {
            "avg_cooperation": self.population.mean().item(),
            "avg_fitness": self.fitness.mean().item(),
            "best_fitness": self.fitness.max().item()
        }
    
    def _calculate_pairwise_payoffs(self, rounds: int) -> torch.Tensor:
        """Calculate payoffs for all pairwise interactions using vectorized operations"""
        # Create all possible pairs
        i, j = torch.meshgrid(torch.arange(self.population_size), 
                             torch.arange(self.population_size), indexing='ij')
        
        # Vectorized payoff calculation
        p1_coop = self.population[i]
        p2_coop = self.population[j]
        
        # Expected payoffs using PyTorch operations
        mutual_coop_prob = p1_coop * p2_coop
        mutual_defect_prob = (1 - p1_coop) * (1 - p2_coop)
        p1_coop_p2_defect_prob = p1_coop * (1 - p2_coop)
        p1_defect_p2_coop_prob = (1 - p1_coop) * p2_coop
        
        # Expected payoffs
        p1_payoff = (mutual_coop_prob * 3 + 
                    mutual_defect_prob * 1 + 
                    p1_coop_p2_defect_prob * 0 + 
                    p1_defect_p2_coop_prob * 5)
        
        return p1_payoff

class ReinforcementLearningAgent:
    """RL agent using PyTorch for learning optimal strategies"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.q_network = nn.Sequential(
            nn.Linear(4, 16),  # State: [my_last_action, opponent_last_action, my_coop_rate, opponent_coop_rate]
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2)   # Actions: [cooperate, defect]
        )
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.epsilon = 0.1  # Exploration rate
        
    def get_action(self, state: torch.Tensor) -> int:
        """Get action using epsilon-greedy policy"""
        if torch.rand(1) < self.epsilon:
            return torch.randint(0, 2, (1,)).item()  # Random action
        
        with torch.no_grad():
            q_values = self.q_network(state)
            return q_values.argmax().item()
    
    def update(self, state: torch.Tensor, action: int, reward: float, 
               next_state: torch.Tensor, done: bool):
        """Update Q-network using PyTorch"""
        q_values = self.q_network(state)
        q_value = q_values[action]
        
        with torch.no_grad():
            next_q_values = self.q_network(next_state)
            target = reward + (0.9 * next_q_values.max() * (1 - done))
        
        loss = nn.MSELoss()(q_value, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class AdvancedMonteCarloSimulator:
    """Enhanced Monte Carlo simulator with PyTorch optimizations"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
    
    def simulate_with_gpu(self, p1: float, p2: float, rounds: int) -> Dict:
        """Run simulation on GPU for massive speedup"""
        # Move tensors to GPU
        p1_tensor = torch.tensor(p1, device=self.device)
        p2_tensor = torch.tensor(p2, device=self.device)
        
        # Generate actions on GPU
        p1_actions = torch.rand(rounds, device=self.device) < p1_tensor
        p2_actions = torch.rand(rounds, device=self.device) < p2_tensor
        
        # Vectorized payoff calculation on GPU
        payoffs = torch.zeros((rounds, 2), device=self.device)
        
        # Use advanced PyTorch operations
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
        
        # Calculate statistics on GPU
        stats = {
            "player1_avg_payoff": payoffs[:, 0].mean().item(),
            "player2_avg_payoff": payoffs[:, 1].mean().item(),
            "player1_coop_rate": p1_actions.float().mean().item(),
            "player2_coop_rate": p2_actions.float().mean().item(),
            "both_cooperate_rate": both_coop.float().mean().item(),
            "both_defect_rate": both_defect.float().mean().item(),
            "p1_coop_p2_defect_rate": p1_coop_p2_defect.float().mean().item(),
            "p1_defect_p2_coop_rate": p1_defect_p2_coop.float().mean().item()
        }
        
        return stats
    
    def batch_simulate_parallel(self, simulations: List[Dict]) -> List[Dict]:
        """Run multiple simulations in parallel using PyTorch"""
        # Stack all parameters into tensors
        p1_probs = torch.tensor([s['player1_prob'] for s in simulations])
        p2_probs = torch.tensor([s['player2_prob'] for s in simulations])
        rounds_list = [s['rounds'] for s in simulations]
        
        results = []
        for i, rounds in enumerate(rounds_list):
            result = self.simulate_with_gpu(p1_probs[i].item(), p2_probs[i].item(), rounds)
            results.append(result)
        
        return results

# Example usage functions
def demonstrate_neural_strategies():
    """Demonstrate neural network-based strategies"""
    player = NeuralStrategyPlayer()
    
    # Simulate with neural strategy
    game_history = torch.rand(10)  # Random history
    coop_prob = player.get_cooperation_prob(game_history)
    
    print(f"Neural strategy cooperation probability: {coop_prob:.3f}")
    return player

def demonstrate_evolutionary_simulation():
    """Demonstrate evolutionary game theory"""
    simulator = EvolutionarySimulator(population_size=50)
    
    print("Evolutionary Simulation:")
    for generation in range(10):
        stats = simulator.evolve_generation(rounds=1000)
        print(f"Generation {generation + 1}: "
              f"Avg Cooperation: {stats['avg_cooperation']:.3f}, "
              f"Avg Fitness: {stats['avg_fitness']:.3f}")
    
    return simulator

def demonstrate_gpu_acceleration():
    """Demonstrate GPU acceleration for large simulations"""
    simulator = AdvancedMonteCarloSimulator(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Large simulation
    result = simulator.simulate_with_gpu(0.5, 0.5, 1_000_000)
    print(f"GPU Simulation Results: {result}")
    
    return simulator

if __name__ == "__main__":
    print("PyTorch Advanced Features Demo")
    print("=" * 40)
    
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Run demonstrations
    neural_player = demonstrate_neural_strategies()
    evolutionary_sim = demonstrate_evolutionary_simulation()
    gpu_simulator = demonstrate_gpu_acceleration()

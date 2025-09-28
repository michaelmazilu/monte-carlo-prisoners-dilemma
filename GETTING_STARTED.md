# 🎯 Getting Started Guide: Monte Carlo Prisoner's Dilemma

## 🚀 **Phase 1: Basic Usage (Start Here!)**

### **1. Test Your Current Setup**
```bash
# Terminal 1: Start Flask backend
cd backend
python app.py

# Terminal 2: Start frontend server  
python -m http.server 8000

# Open browser: http://127.0.0.1:8000
```

### **2. Try Basic Simulations**
- Open the web interface
- Set Player 1 probability: 0.7
- Set Player 2 probability: 0.3  
- Set rounds: 1000
- Click "Run Simulation"
- **Observe**: How cooperation rates affect payoffs

### **3. Test API Directly**
```bash
curl -X POST http://127.0.0.1:5000/simulate \
  -H "Content-Type: application/json" \
  -d '{"player1_prob": 0.8, "player2_prob": 0.2, "rounds": 5000}'
```

## 🔥 **Phase 2: When PyTorch Becomes Essential**

### **Scenario 1: Large Simulations (10K+ rounds)**
**Problem**: Python loops are slow
**Solution**: PyTorch vectorization
```python
# Slow Python way
for i in range(100000):
    action = random.random() < probability

# Fast PyTorch way  
actions = torch.rand(100000) < probability
```

### **Scenario 2: Multiple Strategy Testing**
**Problem**: Testing many parameter combinations
**Solution**: PyTorch batch operations
```python
# Test 100 different probability pairs simultaneously
p1_probs = torch.linspace(0, 1, 100)
p2_probs = torch.linspace(0, 1, 100)
# PyTorch handles all combinations efficiently
```

### **Scenario 3: Statistical Analysis**
**Problem**: Complex calculations on large datasets
**Solution**: PyTorch statistical functions
```python
# Calculate confidence intervals, correlations, etc.
payoffs = torch.tensor(simulation_results)
mean = payoffs.mean()
std = payoffs.std()
confidence_interval = mean + 1.96 * std / torch.sqrt(len(payoffs))
```

## 🧠 **Phase 3: Advanced PyTorch Features**

### **When You Need Neural Networks**
- **Learning Strategies**: Agents that adapt based on opponent behavior
- **Pattern Recognition**: Finding optimal cooperation patterns
- **Prediction**: Forecasting game outcomes

### **When You Need GPU Acceleration**
- **Massive Simulations**: 1M+ rounds
- **Real-time Analysis**: Live strategy optimization
- **Research**: Academic-level simulations

### **When You Need Evolutionary Algorithms**
- **Population Dynamics**: How strategies evolve
- **Genetic Algorithms**: Optimizing strategy parameters
- **Fitness Landscapes**: Understanding cooperation emergence

## 📊 **Practical Learning Exercises**

### **Exercise 1: Basic PyTorch (30 minutes)**
```python
# Run this in Python console
import torch

# Create a simple simulation
rounds = 1000
p1_prob = 0.6
p2_prob = 0.4

# Generate actions (this is what PyTorch does for you)
p1_actions = torch.rand(rounds) < p1_prob
p2_actions = torch.rand(rounds) < p2_prob

# Calculate payoffs
payoffs = torch.zeros((rounds, 2))
payoffs[:, 0] = torch.where(
    p1_actions & p2_actions, 3,  # Both cooperate
    torch.where(p1_actions & ~p2_actions, 0,  # P1 cooperates, P2 defects
               torch.where(~p1_actions & p2_actions, 5, 1))  # P1 defects, P2 cooperates; Both defect
)

print(f"Player 1 average payoff: {payoffs[:, 0].mean():.3f}")
print(f"Player 2 average payoff: {payoffs[:, 1].mean():.3f}")
```

### **Exercise 2: Compare Speed (15 minutes)**
```python
import time
import random

# Python way
start = time.time()
python_actions = [random.random() < 0.5 for _ in range(100000)]
python_time = time.time() - start

# PyTorch way
start = time.time()
torch_actions = torch.rand(100000) < 0.5
torch_time = time.time() - start

print(f"Python: {python_time:.3f}s")
print(f"PyTorch: {torch_time:.3f}s")
print(f"Speedup: {python_time/torch_time:.1f}x")
```

### **Exercise 3: Parameter Sweep (20 minutes)**
```python
# Test different cooperation probabilities
probabilities = torch.linspace(0, 1, 21)  # 0.0 to 1.0 in steps of 0.05
results = []

for p1 in probabilities:
    for p2 in probabilities:
        # Run simulation (use your existing function)
        result = run_monte_carlo_simulation(p1.item(), p2.item(), 1000)
        results.append({
            'p1': p1.item(),
            'p2': p2.item(), 
            'payoff1': result['player1_avg_payoff'],
            'payoff2': result['player2_avg_payoff']
        })

# Find optimal strategies
best_result = max(results, key=lambda x: x['payoff1'] + x['payoff2'])
print(f"Best strategy: P1={best_result['p1']:.2f}, P2={best_result['p2']:.2f}")
```

## 🎯 **When to Upgrade to Advanced PyTorch**

### **Upgrade When You Need:**
1. **Speed**: Simulations taking >30 seconds
2. **Scale**: Testing >10,000 parameter combinations  
3. **Learning**: Want agents that adapt strategies
4. **Research**: Publishing academic papers
5. **GPU**: Have CUDA-capable hardware

### **Don't Upgrade Yet If:**
1. **Learning**: Still understanding basic concepts
2. **Small Scale**: <1,000 rounds work fine
3. **Simple Analysis**: Basic statistics are sufficient
4. **Prototyping**: Rapid iteration is more important

## 🛠️ **Next Steps Based on Your Goals**

### **If You Want to Learn Game Theory:**
- Focus on understanding payoff matrices
- Experiment with different strategy combinations
- Study cooperation vs. defection dynamics

### **If You Want to Learn PyTorch:**
- Start with tensor operations
- Practice vectorization techniques
- Build up to neural networks

### **If You Want to Do Research:**
- Implement evolutionary algorithms
- Add machine learning components
- Scale to massive simulations

### **If You Want to Build Applications:**
- Enhance the web interface
- Add real-time visualization
- Create interactive demos

## 🎮 **Quick Start Commands**

```bash
# 1. Start everything
cd backend && python app.py &
cd .. && python -m http.server 8000 &

# 2. Open browser: http://127.0.0.1:8000

# 3. Test API
curl -X POST http://127.0.0.1:5000/simulate \
  -H "Content-Type: application/json" \
  -d '{"player1_prob": 0.5, "player2_prob": 0.5, "rounds": 1000}'

# 4. Run tests
python test_simulation.py
```

## 📈 **Progression Timeline**

- **Week 1**: Master basic simulations and web interface
- **Week 2**: Understand PyTorch tensor operations  
- **Week 3**: Implement parameter sweeps and analysis
- **Week 4**: Add neural network strategies
- **Month 2**: Build evolutionary algorithms
- **Month 3**: GPU acceleration and research applications

Start simple, build complexity gradually! 🚀

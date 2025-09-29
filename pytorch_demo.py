#!/usr/bin/env python3
"""
🎯 PyTorch Demo for Prisoner's Dilemma
Run this to see PyTorch in action!
"""

import torch
import time
import random
import numpy as np

def demo_basic_pytorch():
    """Demonstrate basic PyTorch operations"""
    print("🔥 Basic PyTorch Operations Demo")
    print("=" * 40)
    
    # 1. Create tensors
    print("1. Creating tensors:")
    rounds = 10000
    p1_prob = 0.6
    p2_prob = 0.4
    
    print(f"   Simulating {rounds:,} rounds")
    print(f"   Player 1 cooperation probability: {p1_prob}")
    print(f"   Player 2 cooperation probability: {p2_prob}")
    
    # 2. Generate random actions (this is where PyTorch shines!)
    print("\n2. Generating actions:")
    start_time = time.time()
    
    p1_actions = torch.rand(rounds) < p1_prob
    p2_actions = torch.rand(rounds) < p2_prob
    
    generation_time = time.time() - start_time
    print(f"   Generated {rounds:,} actions in {generation_time:.4f} seconds")
    print(f"   Player 1 cooperation rate: {p1_actions.float().mean():.3f}")
    print(f"   Player 2 cooperation rate: {p2_actions.float().mean():.3f}")
    
    # 3. Calculate payoffs using vectorized operations
    print("\n3. Calculating payoffs:")
    start_time = time.time()
    
    payoffs = torch.zeros((rounds, 2))
    
    # Both cooperate: (3, 3)
    both_coop = p1_actions & p2_actions
    payoffs[both_coop, 0] = 3
    payoffs[both_coop, 1] = 3
    
    # Both defect: (1, 1)  
    both_defect = ~p1_actions & ~p2_actions
    payoffs[both_defect, 0] = 1
    payoffs[both_defect, 1] = 1
    
    # P1 cooperates, P2 defects: (0, 5)
    p1_coop_p2_defect = p1_actions & ~p2_actions
    payoffs[p1_coop_p2_defect, 0] = 0
    payoffs[p1_coop_p2_defect, 1] = 5
    
    # P1 defects, P2 cooperates: (5, 0)
    p1_defect_p2_coop = ~p1_actions & p2_actions
    payoffs[p1_defect_p2_coop, 0] = 5
    payoffs[p1_defect_p2_coop, 1] = 0
    
    calculation_time = time.time() - start_time
    print(f"   Calculated payoffs in {calculation_time:.4f} seconds")
    
    # 4. Calculate statistics
    print("\n4. Results:")
    print(f"   Player 1 average payoff: {payoffs[:, 0].mean():.3f}")
    print(f"   Player 2 average payoff: {payoffs[:, 1].mean():.3f}")
    print(f"   Both cooperate rate: {both_coop.float().mean():.3f}")
    print(f"   Both defect rate: {both_defect.float().mean():.3f}")
    print(f"   P1 coop, P2 defect rate: {p1_coop_p2_defect.float().mean():.3f}")
    print(f"   P1 defect, P2 coop rate: {p1_defect_p2_coop.float().mean():.3f}")
    
    return payoffs

def demo_speed_comparison():
    """Compare Python vs PyTorch speed"""
    print("\n⚡ Speed Comparison: Python vs PyTorch")
    print("=" * 40)
    
    rounds = 100000
    
    # Python way
    print("Python implementation:")
    start_time = time.time()
    python_actions = [random.random() < 0.5 for _ in range(rounds)]
    python_time = time.time() - start_time
    print(f"   Generated {rounds:,} actions in {python_time:.4f} seconds")
    
    # PyTorch way
    print("PyTorch implementation:")
    start_time = time.time()
    torch_actions = torch.rand(rounds) < 0.5
    torch_time = time.time() - start_time
    print(f"   Generated {rounds:,} actions in {torch_time:.4f} seconds")
    
    speedup = python_time / torch_time
    print(f"\n🚀 PyTorch is {speedup:.1f}x faster!")

def demo_parameter_sweep():
    """Demonstrate parameter sweep with PyTorch"""
    print("\n📊 Parameter Sweep Demo")
    print("=" * 40)
    
    # Test different cooperation probabilities
    probabilities = torch.linspace(0.1, 0.9, 9)  # 0.1 to 0.9
    rounds = 5000
    
    print(f"Testing {len(probabilities)} different probabilities:")
    print("Prob1  Prob2  Avg_Payoff1  Avg_Payoff2  Total_Payoff")
    print("-" * 50)
    
    best_total = 0
    best_combo = None
    
    for p1 in probabilities:
        for p2 in probabilities:
            # Generate actions
            p1_actions = torch.rand(rounds) < p1
            p2_actions = torch.rand(rounds) < p2
            
            # Calculate payoffs
            payoffs = torch.zeros((rounds, 2))
            both_coop = p1_actions & p2_actions
            both_defect = ~p1_actions & ~p2_actions
            p1_coop_p2_defect = p1_actions & ~p2_actions
            p1_defect_p2_coop = ~p1_actions & p2_actions
            
            payoffs[both_coop, 0] = 3
            payoffs[both_coop, 1] = 3
            payoffs[both_defect, 0] = 1
            payoffs[both_defect, 1] = 1
            payoffs[p1_coop_p2_defect, 0] = 0
            payoffs[p1_coop_p2_defect, 1] = 5
            payoffs[p1_defect_p2_coop, 0] = 5
            payoffs[p1_defect_p2_coop, 1] = 0
            
            avg_payoff1 = payoffs[:, 0].mean().item()
            avg_payoff2 = payoffs[:, 1].mean().item()
            total_payoff = avg_payoff1 + avg_payoff2
            
            print(f"{p1:.1f}    {p2:.1f}    {avg_payoff1:.3f}       {avg_payoff2:.3f}       {total_payoff:.3f}")
            
            if total_payoff > best_total:
                best_total = total_payoff
                best_combo = (p1.item(), p2.item())
    
    print(f"\n🏆 Best combination: P1={best_combo[0]:.1f}, P2={best_combo[1]:.1f}")
    print(f"   Total payoff: {best_total:.3f}")

def demo_device_info():
    """Show PyTorch device capabilities"""
    print("\n🖥️  PyTorch Device Information")
    print("=" * 40)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        
        # Test GPU speed
        print("\nTesting GPU vs CPU speed:")
        rounds = 1000000
        
        # CPU
        start_time = time.time()
        cpu_actions = torch.rand(rounds) < 0.5
        cpu_time = time.time() - start_time
        
        # GPU
        start_time = time.time()
        gpu_actions = torch.rand(rounds, device='cuda') < 0.5
        torch.cuda.synchronize()  # Wait for GPU to finish
        gpu_time = time.time() - start_time
        
        print(f"CPU time: {cpu_time:.4f}s")
        print(f"GPU time: {gpu_time:.4f}s")
        print(f"GPU speedup: {cpu_time/gpu_time:.1f}x")
    else:
        print("No CUDA device available - using CPU")

def main():
    """Run all demos"""
    print("PyTorch Prisoner's Dilemma Demo")
    print("=" * 50)
    
    # Run demos
    demo_basic_pytorch()
    demo_speed_comparison()
    demo_parameter_sweep()
    demo_device_info()
    
    print("\n🎉 Demo complete!")
    print("\nNext steps:")
    print("1. Run: python backend/app.py")
    print("2. Open: http://127.0.0.1:5000")
    print("3. Try the web interface!")
    print("4. Experiment with different parameters")

if __name__ == "__main__":
    main()

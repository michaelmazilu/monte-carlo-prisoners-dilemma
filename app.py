from flask import Flask, request, jsonify, Response, render_template_string
from flask_cors import CORS
import torch
import numpy as np
import json
import time
import threading
import queue
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Global variables for real-time updates
simulation_queue = queue.Queue()
current_simulation = None
simulation_thread = None

def run_monte_carlo_simulation(p1, p2, rounds, strategy1="probabilistic", strategy2="probabilistic", random_seed=None):
    """Run Monte Carlo simulation for Prisoner's Dilemma"""
    
    # Set random seed if provided
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
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
        "strategy2": strategy2,
        "parameters": {
            "player1_prob": p1,
            "player2_prob": p2,
            "rounds": rounds,
            "random_seed": random_seed
        }
    }
    
    return result

def run_parameter_sweep_background(rounds_per_config, step_size, progress_callback):
    """Run parameter sweep in background with progress updates"""
    global current_simulation
    
    try:
        # Player 1: 100% → 0% cooperation (decreasing)
        # Player 2: 0% → 100% cooperation (increasing)
        p1_probs = torch.linspace(1.0, 0.0, int(1.0/step_size) + 1)
        p2_probs = torch.linspace(0.0, 1.0, int(1.0/step_size) + 1)
        
        results = []
        total_configs = len(p1_probs) * len(p2_probs)
        
        current_simulation = {
            "status": "running",
            "total_configs": total_configs,
            "completed_configs": 0,
            "start_time": time.time(),
            "results": []
        }
        
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
                current_simulation["completed_configs"] += 1
                current_simulation["results"] = results
                
                # Send progress update
                progress = (current_simulation["completed_configs"] / total_configs) * 100
                progress_callback({
                    "type": "progress",
                    "progress": progress,
                    "completed": current_simulation["completed_configs"],
                    "total": total_configs,
                    "current_config": {
                        "p1_prob": p1.item(),
                        "p2_prob": p2.item(),
                        "result": result
                    }
                })
                
                # Small delay to make it visible
                time.sleep(0.01)
        
        # Send completion
        progress_callback({
            "type": "complete",
            "results": results,
            "summary": {
                "total_configurations": total_configs,
                "rounds_per_config": rounds_per_config,
                "step_size": step_size,
                "p1_range": [p1_probs[0].item(), p1_probs[-1].item()],
                "p2_range": [p2_probs[0].item(), p2_probs[-1].item()],
                "execution_time": time.time() - current_simulation["start_time"]
            }
        })
        
        current_simulation["status"] = "completed"
        
    except Exception as e:
        current_simulation["status"] = "error"
        progress_callback({
            "type": "error",
            "error": str(e)
        })

@app.route("/")
def home():
    """Serve the main application page"""
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Monte Carlo Prisoner's Dilemma Simulator</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --bg-primary: #FFFFFF;
            --bg-secondary: #F9FAFB;
            --bg-tertiary: #F3F4F6;
            --text-primary: #111827;
            --text-secondary: #6B7280;
            --text-muted: #9CA3AF;
            --border-color: #E5E7EB;
            --shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            --chart-bg: #FFFFFF;
        }

        [data-theme="dark"] {
            --bg-primary: #111827;
            --bg-secondary: #1F2937;
            --bg-tertiary: #374151;
            --text-primary: #F9FAFB;
            --text-secondary: #D1D5DB;
            --text-muted: #9CA3AF;
            --border-color: #374151;
            --shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
            --chart-bg: #1F2937;
        }

        html {
            font-size: 16px;
        }

        @media (min-width: 1280px) {
            html { font-size: 17px; }
        }

        @media (min-width: 1536px) {
            html { font-size: 18px; }
        }

        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
            transition: background-color 0.3s ease, color 0.3s ease;
        }

        .container {
            max-width: 1440px;
            margin: 0 auto;
            padding: 24px 32px;
        }

        .header {
            text-align: left;
            margin-bottom: 60px;
            position: relative;
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 8px;
        }

        .header p {
            font-size: 1.125rem;
            font-weight: 400;
            color: var(--text-secondary);
        }

        .theme-toggle {
            position: absolute;
            top: 0;
            right: 0;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 40px;
            height: 40px;
        }

        .theme-toggle:hover {
            background: var(--bg-tertiary);
        }

        .theme-icon {
            width: 20px;
            height: 20px;
            stroke: var(--text-primary);
            fill: none;
            stroke-width: 2;
            stroke-linecap: round;
            stroke-linejoin: round;
        }

        .simulation-controls {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            padding: 40px;
            margin-bottom: 40px;
            text-align: left;
            transition: background-color 0.3s ease, border-color 0.3s ease;
        }

        .parameter-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 24px;
            margin: 32px 0;
        }

        .parameter-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .parameter-group label {
            font-weight: 600;
            color: var(--text-primary);
            font-size: 0.95rem;
        }

        .slider-container {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .slider {
            flex: 1;
            height: 6px;
            border-radius: 3px;
            background: var(--border-color);
            outline: none;
            -webkit-appearance: none;
        }

        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: var(--text-primary);
            cursor: pointer;
        }

        .slider::-moz-range-thumb {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: var(--text-primary);
            cursor: pointer;
            border: none;
        }

        .slider-container span {
            min-width: 60px;
            text-align: right;
            font-weight: 500;
            color: var(--text-primary);
            font-size: 0.875rem;
        }

        .select-input, .number-input {
            padding: 8px 12px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 1rem;
            background: var(--bg-primary);
            color: var(--text-primary);
        }

        .select-input:focus, .number-input:focus {
            outline: none;
            border-color: var(--text-primary);
            box-shadow: 0 0 0 3px rgba(17, 24, 39, 0.1);
        }

        .button-group {
            display: flex;
            gap: 12px;
            margin-top: 32px;
            flex-wrap: wrap;
        }

        .introduction {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            padding: 40px;
            margin-bottom: 40px;
            border-radius: 8px;
            transition: background-color 0.3s ease, border-color 0.3s ease;
        }

        .introduction h2 {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 16px;
        }

        .introduction h3 {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
            margin: 32px 0 16px 0;
        }

        .introduction p {
            font-size: 1rem;
            color: var(--text-secondary);
            line-height: 1.6;
            margin-bottom: 16px;
        }

        .introduction ul, .introduction ol {
            margin: 16px 0;
            padding-left: 24px;
        }

        .introduction li {
            font-size: 1rem;
            color: var(--text-secondary);
            line-height: 1.6;
            margin-bottom: 8px;
        }

        .payoff-matrix {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            padding: 24px;
            border-radius: 8px;
            margin: 24px 0;
            transition: background-color 0.3s ease, border-color 0.3s ease;
        }

        .matrix-container {
            display: flex;
            justify-content: center;
            margin: 20px 0;
        }

        .payoff-table {
            border-collapse: collapse;
            background: var(--bg-primary);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: var(--shadow);
        }

        .payoff-table th, .payoff-table td {
            border: 1px solid var(--border-color);
            padding: 16px;
            text-align: center;
            font-size: 0.875rem;
        }

        .payoff-table th {
            background: var(--text-primary);
            color: var(--bg-primary);
            font-weight: 600;
        }

        .payoff-table td {
            font-weight: 500;
        }

        .cooperate-both {
            background: #D1FAE5;
            color: #065F46;
        }

        .temptation-sucker {
            background: #FEF3C7;
            color: #92400E;
        }

        .sucker-temptation {
            background: #FEE2E2;
            color: #991B1B;
        }

        .defect-both {
            background: #F3F4F6;
            color: #374151;
        }

        .payoff-explanation {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-top: 20px;
        }

        .payoff-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px;
            background: var(--bg-primary);
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }

        .payoff-value {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            font-weight: 600;
            font-size: 0.875rem;
            color: white;
        }

        .payoff-value.cooperate-both {
            background: #10B981;
        }

        .payoff-value.temptation-sucker {
            background: #F59E0B;
        }

        .payoff-value.sucker-temptation {
            background: #EF4444;
        }

        .payoff-value.defect-both {
            background: #6B7280;
        }

        .payoff-label {
            font-size: 0.875rem;
            color: var(--text-primary);
            font-weight: 500;
        }

        .monte-carlo-explanation {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            padding: 24px;
            border-radius: 8px;
            margin: 24px 0;
            transition: background-color 0.3s ease, border-color 0.3s ease;
        }

        .how-to-use {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            padding: 24px;
            border-radius: 8px;
            margin: 24px 0;
            transition: background-color 0.3s ease, border-color 0.3s ease;
        }

        .simulation-controls h2 {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 16px;
        }

        .simulation-controls p {
            font-size: 1rem;
            color: var(--text-secondary);
            margin-bottom: 12px;
        }

        .simulation-controls ul {
            margin: 20px 0 30px 0;
            padding-left: 20px;
        }

        .simulation-controls li {
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }

        .btn {
            font-family: 'Inter', system-ui, sans-serif;
            font-size: 1rem;
            font-weight: 500;
            padding: 14px 26px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            margin-right: 12px;
            margin-bottom: 12px;
        }

        .btn-primary {
            background: var(--text-primary);
            color: var(--bg-primary);
            border: 1px solid var(--text-primary);
        }

        .btn-primary:hover:not(:disabled) {
            background: var(--text-secondary);
            border-color: var(--text-secondary);
        }

        .btn-outline {
            background: transparent;
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
        }

        .btn-outline:hover:not(:disabled) {
            background: var(--bg-secondary);
            color: var(--text-primary);
            border-color: var(--text-muted);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .progress-container {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            padding: 32px;
            margin: 40px 0;
            display: none;
            transition: background-color 0.3s ease, border-color 0.3s ease;
        }

        .progress-container h3 {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 24px;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: var(--border-color);
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 16px;
        }

        .progress-fill {
            height: 100%;
            background: var(--text-secondary);
            width: 0%;
            transition: width 0.3s ease;
        }

        .progress-text {
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-primary);
            margin-bottom: 20px;
        }

        .config-display {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            padding: 16px;
            border-radius: 6px;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        .stats-grid {
            display: flex;
            gap: 24px;
            margin: 40px 0;
            flex-wrap: wrap;
        }

        .stat-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .stat-label {
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .stat-value {
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .charts-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 32px;
            margin-top: 40px;
        }

        .histogram-container {
            display: grid;
            grid-template-columns: 1fr;
            gap: 32px;
            margin-top: 40px;
        }

        .chart-container {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            padding: 24px;
            border-radius: 8px;
            transition: background-color 0.3s ease, border-color 0.3s ease;
        }

        .chart-container canvas {
            height: 420px !important;
        }

        .live-updates {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            padding: 24px;
            margin: 40px 0;
            max-height: 240px;
            overflow-y: auto;
            border-radius: 8px;
            transition: background-color 0.3s ease, border-color 0.3s ease;
        }

        .live-updates h4 {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 16px;
        }

        .update-item {
            padding: 8px 0;
            border-bottom: 1px solid var(--border-color);
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        .update-item:last-child {
            border-bottom: none;
        }

        .status-indicator {
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            margin-right: 8px;
        }

        .status-running {
            background: #10B981;
            animation: pulse 1.5s infinite;
        }

        .status-completed {
            background: #6B7280;
        }

        .status-error {
            background: #EF4444;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .container {
                padding: 20px 16px;
            }
            
            .charts-container {
                grid-template-columns: 1fr;
                gap: 24px;
            }
            
            .stats-grid {
                flex-direction: column;
                gap: 16px;
            }
            
            .simulation-controls {
                padding: 24px;
            }
        }

        /* Footer and modal for explanations */
        .explain-footer {
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border-color);
            padding: 12px 16px;
            display: flex;
            justify-content: center;
            gap: 12px;
            z-index: 50;
        }

        .explain-modal-overlay {
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.4);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 60;
        }

        .explain-modal {
            width: min(900px, 92vw);
            max-height: 85vh;
            overflow: auto;
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            box-shadow: var(--shadow);
        }

        .explain-modal-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
        }

        .explain-modal-title {
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .explain-modal-close {
            background: transparent;
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 6px 10px;
            border-radius: 6px;
            cursor: pointer;
        }

        .explain-modal-body {
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <button class="theme-toggle" onclick="toggleTheme()" id="themeToggle">
                <svg class="theme-icon" id="themeIcon" viewBox="0 0 24 24">
                    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                </svg>
            </button>
            <h1>Monte Carlo Prisoner's Dilemma Simulator</h1>
            <p>Real-time parameter sweep with 10,000 experiments</p>
        </div>

        <div class="introduction">
            <h2>What is the Prisoner's Dilemma?</h2>
            <p>The Prisoner's Dilemma is a classic game theory scenario where two players must choose between <strong>cooperating</strong> or <strong>defecting</strong> without knowing the other's choice. The outcome depends on both players' decisions.</p>
            
            <div class="payoff-matrix">
                <h3>Payoff Matrix</h3>
                <div class="matrix-container">
                    <table class="payoff-table">
                        <thead>
                            <tr>
                                <th></th>
                                <th>Player 2: Cooperate</th>
                                <th>Player 2: Defect</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <th>Player 1: Cooperate</th>
                                <td class="cooperate-both">(3, 3)<br><small>Both get 3 points</small></td>
                                <td class="sucker-temptation">(0, 5)<br><small>P1: 0, P2: 5</small></td>
                            </tr>
                            <tr>
                                <th>Player 1: Defect</th>
                                <td class="temptation-sucker">(5, 0)<br><small>P1: 5, P2: 0</small></td>
                                <td class="defect-both">(1, 1)<br><small>Both get 1 point</small></td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                <div class="payoff-explanation">
                    <div class="payoff-item">
                        <span class="payoff-value cooperate-both">3</span>
                        <span class="payoff-label">Mutual Cooperation - Both players benefit</span>
                    </div>
                    <div class="payoff-item">
                        <span class="payoff-value temptation-sucker">5</span>
                        <span class="payoff-label">Temptation - Best individual outcome</span>
                    </div>
                    <div class="payoff-item">
                        <span class="payoff-value sucker-temptation">0</span>
                        <span class="payoff-label">Sucker's Payoff - Worst individual outcome</span>
                    </div>
                    <div class="payoff-item">
                        <span class="payoff-value defect-both">1</span>
                        <span class="payoff-label">Mutual Defection - Both players lose</span>
                    </div>
                </div>
            </div>

            <div class="monte-carlo-explanation">
                <h3>What is Monte Carlo Simulation?</h3>
                <p>Monte Carlo simulation runs thousands of random trials to estimate outcomes. In this context:</p>
                <ul>
                    <li><strong>Random Trials:</strong> Each "round" is a random game where players choose based on their cooperation probability</li>
                    <li><strong>Statistical Accuracy:</strong> Running 1,000+ rounds gives us reliable average payoffs</li>
                    <li><strong>Strategy Analysis:</strong> We can test how different cooperation rates affect outcomes</li>
                    <li><strong>Parameter Sweeps:</strong> We can test all combinations of cooperation probabilities (0% to 100%)</li>
                </ul>
            </div>

            <div class="how-to-use">
                <h3>🚀 How to Use This Simulator</h3>
                <ol>
                    <li><strong>Set Parameters:</strong> Adjust cooperation probabilities, number of rounds, and strategies</li>
                    <li><strong>Run Custom Simulation:</strong> Test specific parameter combinations</li>
                    <li><strong>Run Parameter Sweep:</strong> Analyze all 10,000 combinations automatically</li>
                    <li><strong>View Results:</strong> See charts showing payoffs, cooperation rates, and outcome distributions</li>
                </ol>
            </div>
        </div>

        <div class="simulation-controls">
            <h2>Custom Simulation Parameters</h2>
            <p>Configure your simulation parameters:</p>
            
            <div class="parameter-grid">
                <div class="parameter-group">
                    <label for="player1_prob">Player 1 Cooperation Probability</label>
                    <div class="slider-container">
                        <input type="range" id="player1_prob" min="0" max="100" value="50" class="slider">
                        <span id="player1_value">50%</span>
                    </div>
                </div>
                
                <div class="parameter-group">
                    <label for="player2_prob">Player 2 Cooperation Probability</label>
                    <div class="slider-container">
                        <input type="range" id="player2_prob" min="0" max="100" value="50" class="slider">
                        <span id="player2_value">50%</span>
                    </div>
                </div>
                
                <div class="parameter-group">
                    <label for="rounds">Number of Rounds</label>
                    <div class="slider-container">
                        <input type="range" id="rounds" min="100" max="100000" value="1000" step="100" class="slider">
                        <span id="rounds_value">1,000</span>
                    </div>
                </div>
                
                <div class="parameter-group">
                    <label for="strategy1">Player 1 Strategy</label>
                    <select id="strategy1" class="select-input">
                        <option value="probabilistic">Probabilistic</option>
                        <option value="always_cooperate">Always Cooperate</option>
                        <option value="always_defect">Always Defect</option>
                    </select>
                </div>
                
                <div class="parameter-group">
                    <label for="strategy2">Player 2 Strategy</label>
                    <select id="strategy2" class="select-input">
                        <option value="probabilistic">Probabilistic</option>
                        <option value="always_cooperate">Always Cooperate</option>
                        <option value="always_defect">Always Defect</option>
                    </select>
                </div>
                
                <div class="parameter-group">
                    <label for="random_seed">Random Seed (optional)</label>
                    <input type="number" id="random_seed" placeholder="Leave empty for random" class="number-input">
                </div>
            </div>
            
            <div class="button-group">
                <button class="btn btn-primary" id="runCustomBtn" onclick="runCustomSimulation()">Run Custom Simulation</button>
                <button class="btn btn-outline" id="runSweepBtn" onclick="startSimulation()">Run Parameter Sweep (10,000 experiments)</button>
            </div>
        </div>

        <div class="progress-container" id="progressContainer">
            <h3>Simulation Progress</h3>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <div class="progress-text" id="progressText">Ready to start...</div>
            
            <div class="config-display" id="currentConfig">
                <strong>Current Configuration:</strong> <span id="configText">Waiting...</span>
            </div>
        </div>

        <div class="stats-grid" id="statsGrid" style="display: none;">
            <div class="stat-item">
                <span class="stat-label">Completed:</span>
                <span class="stat-value" id="completedCount">0</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Total:</span>
                <span class="stat-value" id="totalCount">10,000</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Progress:</span>
                <span class="stat-value" id="progressPercent">0%</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Elapsed:</span>
                <span class="stat-value" id="elapsedTime">0s</span>
            </div>
        </div>

        <div class="live-updates" id="liveUpdates">
            <h4>Live Updates</h4>
            <div id="updatesList">
                <div class="update-item">
                    <span class="status-indicator status-completed"></span>
                    Ready to start simulation...
                </div>
            </div>
        </div>

        <div class="charts-container" id="chartsContainer" style="display: none;">
            <div class="chart-container">
                <canvas id="p1Chart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="p2Chart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="totalChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="coopChart"></canvas>
            </div>
        </div>

        <div class="histogram-container" id="histogramContainer" style="display: none;">
            <div class="chart-container" style="grid-column: 1 / -1;">
                <canvas id="histogramChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        let eventSource = null;
        let startTime = null;
        let charts = {};

        // Initialize parameter controls
        document.addEventListener('DOMContentLoaded', () => {
            // Set up slider value updates
            const player1Slider = document.getElementById('player1_prob');
            const player2Slider = document.getElementById('player2_prob');
            const roundsSlider = document.getElementById('rounds');
            
            const player1Value = document.getElementById('player1_value');
            const player2Value = document.getElementById('player2_value');
            const roundsValue = document.getElementById('rounds_value');
            
            player1Slider.addEventListener('input', () => {
                player1Value.textContent = player1Slider.value + '%';
            });
            
            player2Slider.addEventListener('input', () => {
                player2Value.textContent = player2Slider.value + '%';
            });
            
            roundsSlider.addEventListener('input', () => {
                const value = parseInt(roundsSlider.value);
                roundsValue.textContent = value.toLocaleString();
            });
        });

        function runCustomSimulation() {
            const runCustomBtn = document.getElementById('runCustomBtn');
            const progressContainer = document.getElementById('progressContainer');
            const statsGrid = document.getElementById('statsGrid');
            const chartsContainer = document.getElementById('chartsContainer');
            const histogramContainer = document.getElementById('histogramContainer');
            
            // Get parameter values
            const player1Prob = parseFloat(document.getElementById('player1_prob').value) / 100;
            const player2Prob = parseFloat(document.getElementById('player2_prob').value) / 100;
            const rounds = parseInt(document.getElementById('rounds').value);
            const strategy1 = document.getElementById('strategy1').value;
            const strategy2 = document.getElementById('strategy2').value;
            const randomSeed = document.getElementById('random_seed').value;
            
            // Validate parameters
            if (rounds < 100 || rounds > 100000) {
                alert('Number of rounds must be between 100 and 100,000');
                return;
            }
            
            runCustomBtn.disabled = true;
            progressContainer.style.display = 'block';
            statsGrid.style.display = 'flex';
            chartsContainer.style.display = 'grid';
            histogramContainer.style.display = 'grid';
            
            // Show loading state
            document.getElementById('progressText').textContent = 'Running custom simulation...';
            document.getElementById('progressFill').style.width = '0%';
            
            // Prepare request data
            const requestData = {
                player1_prob: player1Prob,
                player2_prob: player2Prob,
                rounds: rounds,
                strategy1: strategy1,
                strategy2: strategy2
            };
            
            if (randomSeed) {
                requestData.random_seed = parseInt(randomSeed);
            }
            
            // Send request to backend
            fetch('/simulate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Update progress to 100%
                document.getElementById('progressFill').style.width = '100%';
                document.getElementById('progressText').textContent = 'Simulation completed!';
                
                // Display results
                displayCustomResults(data);
                
                runCustomBtn.disabled = false;
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Simulation failed: ' + error.message);
                runCustomBtn.disabled = false;
                progressContainer.style.display = 'none';
            });
        }

        function startSimulation() {
            const runSweepBtn = document.getElementById('runSweepBtn');
            const progressContainer = document.getElementById('progressContainer');
            const statsGrid = document.getElementById('statsGrid');
            const chartsContainer = document.getElementById('chartsContainer');
            const histogramContainer = document.getElementById('histogramContainer');
            
            runSweepBtn.disabled = true;
            progressContainer.style.display = 'block';
            statsGrid.style.display = 'flex';
            chartsContainer.style.display = 'grid';
            histogramContainer.style.display = 'grid';
            
            // Reset chart data
            chartData = {
                p1Payoffs: [],
                p2Payoffs: [],
                totalPayoffs: [],
                cooperationRates: [],
                labels: []
            };
            
            // Create empty charts immediately for real-time updates
            charts.p1 = createChart('p1Chart', [], 'Player 1 Average Payoffs', '#6B7280');
            charts.p2 = createChart('p2Chart', [], 'Player 2 Average Payoffs', '#9CA3AF');
            charts.total = createChart('totalChart', [], 'Total Average Payoffs', '#6B7280');
            charts.coop = createChart('coopChart', [], 'Cooperation Rates', '#9CA3AF');
            
            // Create histogram chart
            charts.histogram = createHistogramChart();
            
            startTime = Date.now();
            
            // Start Server-Sent Events connection
            eventSource = new EventSource('/simulation_stream');
            
            // Add connection monitoring
            let lastUpdateTime = Date.now();
            const connectionMonitor = setInterval(() => {
                if (Date.now() - lastUpdateTime > 30000) { // 30 seconds timeout
                    console.warn('Connection timeout detected');
                    addUpdate('Connection timeout - resetting', 'error');
                    
                    // Reset UI
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                    
                    if (eventSource) {
                        eventSource.close();
                        eventSource = null;
                    }
                    
                    clearInterval(connectionMonitor);
                }
            }, 5000);
            
            eventSource.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    handleSimulationUpdate(data);
                } catch (error) {
                    console.error('Error parsing SSE data:', error);
                    addUpdate('Error processing simulation data', 'error');
                }
            };
            
            eventSource.onerror = function(event) {
                console.error('SSE error:', event);
                addUpdate('Connection error occurred', 'error');
                
                // Reset UI state
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                
                // Close connection
                if (eventSource) {
                    eventSource.close();
                    eventSource = null;
                }
            };
            
            // Start the simulation
            fetch('/start_simulation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    rounds_per_config: 100,
                    step_size: 0.01
                })
            }).catch(error => {
                console.error('Error starting simulation:', error);
                addUpdate('Failed to start simulation', 'error');
            });
        }

        function stopSimulation() {
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
            
            fetch('/stop_simulation', { method: 'POST' });
            
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            
            addUpdate('Simulation stopped by user', 'error');
        }

        function handleSimulationUpdate(data) {
            try {
                switch(data.type) {
                    case 'progress':
                        updateProgress(data);
                        break;
                    case 'complete':
                        handleCompletion(data);
                        break;
                    case 'error':
                        handleError(data);
                        break;
                    case 'ping':
                        // Keep-alive message, do nothing
                        break;
                    default:
                        console.warn('Unknown message type:', data.type);
                }
            } catch (error) {
                console.error('Error handling simulation update:', error);
                addUpdate('Error processing update: ' + error.message, 'error');
                
                // Reset UI state on error
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                
                if (eventSource) {
                    eventSource.close();
                    eventSource = null;
                }
            }
        }

        function updateProgress(data) {
            const progressFill = document.getElementById('progressFill');
            const progressText = document.getElementById('progressText');
            const completedCount = document.getElementById('completedCount');
            const progressPercent = document.getElementById('progressPercent');
            const elapsedTime = document.getElementById('elapsedTime');
            const configText = document.getElementById('configText');
            
            // Fix progress bar - use actual progress percentage
            progressFill.style.width = data.progress + '%';
            progressText.textContent = `Processing ${Math.round(data.progress)}% of configurations...`;
            completedCount.textContent = data.completed.toLocaleString();
            progressPercent.textContent = Math.round(data.progress) + '%';
            
            if (startTime) {
                const elapsed = Math.round((Date.now() - startTime) / 1000);
                elapsedTime.textContent = elapsed + 's';
            }
            
            if (data.current_config) {
                const config = data.current_config;
                configText.textContent = `P1: ${(config.p1_prob * 100).toFixed(0)}%, P2: ${(config.p2_prob * 100).toFixed(0)}% | P1 Payoff: ${config.result.player1_avg_payoff.toFixed(2)}, P2 Payoff: ${config.result.player2_avg_payoff.toFixed(2)}`;
                
                // Update charts in real-time
                updateChartsRealTime(config.result, data.completed);
            }
            
            addUpdate(`Completed ${data.completed}/${data.total} configurations (${Math.round(data.progress)}%)`, 'running');
        }

        function handleCompletion(data) {
            addUpdate('Simulation completed successfully!', 'completed');
            
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
            
            // Create charts with final results
            createCharts(data.results);
            document.getElementById('chartsContainer').style.display = 'grid';
        }

        function handleError(data) {
            addUpdate('Error: ' + data.error, 'error');
            
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
        }

        function addUpdate(message, type) {
            const updatesList = document.getElementById('updatesList');
            const updateItem = document.createElement('div');
            updateItem.className = 'update-item';
            
            const timestamp = new Date().toLocaleTimeString();
            updateItem.innerHTML = `
                <span class="status-indicator status-${type}"></span>
                [${timestamp}] ${message}
            `;
            
            updatesList.insertBefore(updateItem, updatesList.firstChild);
            
            // Keep only last 20 updates
            while (updatesList.children.length > 20) {
                updatesList.removeChild(updatesList.lastChild);
            }
        }

        // Real-time chart data storage
        let chartData = {
            p1Payoffs: [],
            p2Payoffs: [],
            totalPayoffs: [],
            cooperationRates: [],
            labels: []
        };

        // Histogram data storage
        let histogramData = {
            cooperationLevels: [], // 0% to 100%
            p1TotalCoins: [], // Total coins for each cooperation level
            p2TotalCoins: []  // Total coins for each cooperation level
        };

        function displayCustomResults(result) {
            // Update statistics display
            document.getElementById('completedCount').textContent = '1';
            document.getElementById('totalCount').textContent = '1';
            document.getElementById('progressPercent').textContent = '100%';
            
            // Create single result charts
            createCustomCharts(result);
        }

        function createCustomCharts(result) {
            // Clear existing charts
            if (charts.p1) charts.p1.destroy();
            if (charts.p2) charts.p2.destroy();
            if (charts.total) charts.total.destroy();
            if (charts.coop) charts.coop.destroy();
            
            // Create bar chart for payoffs
            charts.p1 = createBarChart('p1Chart', 
                ['Player 1', 'Player 2'], 
                [result.player1_avg_payoff, result.player2_avg_payoff],
                'Average Payoffs by Player',
                ['#6B7280', '#9CA3AF']
            );
            
            // Create pie chart for cooperation outcomes
            charts.coop = createPieChart('coopChart',
                ['Both Cooperate', 'Both Defect', 'P1 Coop, P2 Defect', 'P1 Defect, P2 Coop'],
                [
                    result.both_cooperate_rate * 100,
                    result.both_defect_rate * 100,
                    result.p1_coop_p2_defect_rate * 100,
                    result.p1_defect_p2_coop_rate * 100
                ],
                'Outcome Distribution (%)',
                ['#10B981', '#EF4444', '#F59E0B', '#8B5CF6']
            );
            
            // Create line chart for total payoff
            charts.total = createLineChart('totalChart',
                ['Total Payoff'],
                [result.player1_avg_payoff + result.player2_avg_payoff],
                'Total Average Payoff',
                '#6B7280'
            );
            
            // Create cooperation rate chart
            charts.p2 = createLineChart('p2Chart',
                ['Cooperation Rate'],
                [result.both_cooperate_rate * 100],
                'Mutual Cooperation Rate (%)',
                '#9CA3AF'
            );
        }

        function createBarChart(canvasId, labels, data, title, colors) {
            const ctx = document.getElementById(canvasId).getContext('2d');
            return new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        data: data,
                        backgroundColor: colors,
                        borderColor: colors,
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        title: {
                            display: true,
                            text: title,
                            font: { family: 'Inter, system-ui, sans-serif', size: 14, weight: '500' },
                            color: '#374151'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { display: true, color: '#F3F4F6' },
                            ticks: { color: '#6B7280', font: { family: 'Inter, system-ui, sans-serif', size: 11 } }
                        },
                        x: {
                            grid: { display: true, color: '#F3F4F6' },
                            ticks: { color: '#6B7280', font: { family: 'Inter, system-ui, sans-serif', size: 11 } }
                        }
                    }
                }
            });
        }

        function createPieChart(canvasId, labels, data, title, colors) {
            const ctx = document.getElementById(canvasId).getContext('2d');
            return new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{
                        data: data,
                        backgroundColor: colors,
                        borderWidth: 2,
                        borderColor: '#fff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                usePointStyle: true,
                                padding: 20,
                                font: { family: 'Inter, system-ui, sans-serif', size: 12, weight: '500' },
                                color: '#374151'
                            }
                        },
                        title: {
                            display: true,
                            text: title,
                            font: { family: 'Inter, system-ui, sans-serif', size: 14, weight: '500' },
                            color: '#374151'
                        }
                    }
                }
            });
        }

        function createLineChart(canvasId, labels, data, title, color) {
            const ctx = document.getElementById(canvasId).getContext('2d');
            return new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        data: data,
                        borderColor: color,
                        backgroundColor: color + '15',
                        fill: false,
                        tension: 0.1,
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        borderWidth: 3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        title: {
                            display: true,
                            text: title,
                            font: { family: 'Inter, system-ui, sans-serif', size: 14, weight: '500' },
                            color: '#374151'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { display: true, color: '#F3F4F6' },
                            ticks: { color: '#6B7280', font: { family: 'Inter, system-ui, sans-serif', size: 11 } }
                        },
                        x: {
                            grid: { display: true, color: '#F3F4F6' },
                            ticks: { color: '#6B7280', font: { family: 'Inter, system-ui, sans-serif', size: 11 } }
                        }
                    }
                }
            });
        }

        function updateChartsRealTime(result, completed) {
            // Add new data point
            chartData.p1Payoffs.push(result.player1_avg_payoff);
            chartData.p2Payoffs.push(result.player2_avg_payoff);
            chartData.totalPayoffs.push(result.player1_avg_payoff + result.player2_avg_payoff);
            chartData.cooperationRates.push(result.both_cooperate_rate);
            chartData.labels.push(completed);
            
            // Update charts if they exist
            if (charts.p1) {
                charts.p1.data.datasets[0].data = chartData.p1Payoffs;
                charts.p1.data.labels = chartData.labels;
                charts.p1.update('none');
            }
            if (charts.p2) {
                charts.p2.data.datasets[0].data = chartData.p2Payoffs;
                charts.p2.data.labels = chartData.labels;
                charts.p2.update('none');
            }
            if (charts.total) {
                charts.total.data.datasets[0].data = chartData.totalPayoffs;
                charts.total.data.labels = chartData.labels;
                charts.total.update('none');
            }
            if (charts.coop) {
                charts.coop.data.datasets[0].data = chartData.cooperationRates;
                charts.coop.data.labels = chartData.labels;
                charts.coop.update('none');
            }
            
            // Update histogram data
            updateHistogramData(result);
        }

        function createCharts(results) {
            // Prepare data
            chartData.p1Payoffs = results.map(r => r.player1_avg_payoff);
            chartData.p2Payoffs = results.map(r => r.player2_avg_payoff);
            chartData.totalPayoffs = results.map(r => r.player1_avg_payoff + r.player2_avg_payoff);
            chartData.cooperationRates = results.map(r => r.both_cooperate_rate);
            chartData.labels = results.map((_, i) => i + 1);
            
            // Create charts
            charts.p1 = createChart('p1Chart', chartData.p1Payoffs, 'Player 1 Average Payoffs', '#6B7280');
            charts.p2 = createChart('p2Chart', chartData.p2Payoffs, 'Player 2 Average Payoffs', '#9CA3AF');
            charts.total = createChart('totalChart', chartData.totalPayoffs, 'Total Average Payoffs', '#6B7280');
            charts.coop = createChart('coopChart', chartData.cooperationRates, 'Cooperation Rates', '#9CA3AF');
        }

        function createChart(canvasId, data, title, color) {
            const ctx = document.getElementById(canvasId).getContext('2d');
            
            return new Chart(ctx, {
                type: 'line',
                data: {
                    labels: chartData.labels,
                    datasets: [{
                        data: data,
                        borderColor: color,
                        backgroundColor: color + '15',
                        fill: false,
                        tension: 0.1,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        },
                        title: {
                            display: true,
                            text: title,
                            font: {
                                family: 'Inter, system-ui, sans-serif',
                                size: 14,
                                weight: '500'
                            },
                            color: '#374151'
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            grid: {
                                display: true,
                                color: '#F3F4F6',
                                drawBorder: false
                            },
                            ticks: {
                                color: '#6B7280',
                                font: {
                                    family: 'Inter, system-ui, sans-serif',
                                    size: 11
                                }
                            }
                        },
                        y: {
                            display: true,
                            beginAtZero: true,
                            grid: {
                                display: true,
                                color: '#F3F4F6',
                                drawBorder: false
                            },
                            ticks: {
                                color: '#6B7280',
                                font: {
                                    family: 'Inter, system-ui, sans-serif',
                                    size: 11
                                }
                            }
                        }
                    },
                    elements: {
                        point: {
                            hoverBackgroundColor: color
                        }
                    }
                }
            });
        }

        function updateHistogramData(result) {
            // Get cooperation levels from the result
            const p1CoopLevel = Math.round(result.p1_prob * 100);
            const p2CoopLevel = Math.round(result.p2_prob * 100);
            
            // Initialize histogram data if empty
            if (histogramData.cooperationLevels.length === 0) {
                for (let i = 0; i <= 100; i++) {
                    histogramData.cooperationLevels.push(i);
                    histogramData.p1TotalCoins.push(0);
                    histogramData.p2TotalCoins.push(0);
                }
            }
            
            // Add coins to the appropriate cooperation levels
            histogramData.p1TotalCoins[p1CoopLevel] += result.player1_avg_payoff;
            histogramData.p2TotalCoins[p2CoopLevel] += result.player2_avg_payoff;
            
            // Update histogram chart
            if (charts.histogram) {
                charts.histogram.data.datasets[0].data = histogramData.p1TotalCoins;
                charts.histogram.data.datasets[1].data = histogramData.p2TotalCoins;
                charts.histogram.update('none');
            }
        }

        function createHistogramChart() {
            const ctx = document.getElementById('histogramChart').getContext('2d');
            
            // Initialize histogram data
            histogramData.cooperationLevels = [];
            histogramData.p1TotalCoins = [];
            histogramData.p2TotalCoins = [];
            
            for (let i = 0; i <= 100; i++) {
                histogramData.cooperationLevels.push(i);
                histogramData.p1TotalCoins.push(0);
                histogramData.p2TotalCoins.push(0);
            }
            
            return new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: histogramData.cooperationLevels,
                    datasets: [
                        {
                            label: 'Player 1 Total Coins',
                            data: histogramData.p1TotalCoins,
                            backgroundColor: '#3B82F6',
                            borderColor: '#3B82F6',
                            borderWidth: 0,
                            barThickness: 8
                        },
                        {
                            label: 'Player 2 Total Coins',
                            data: histogramData.p2TotalCoins,
                            backgroundColor: '#EF4444',
                            borderColor: '#EF4444',
                            borderWidth: 0,
                            barThickness: 8
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top',
                            labels: {
                                usePointStyle: true,
                                padding: 20,
                                font: {
                                    family: 'Inter, system-ui, sans-serif',
                                    size: 12,
                                    weight: '500'
                                },
                                color: '#374151'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Payoff Distribution by Cooperation Probability',
                            font: {
                                family: 'Inter, system-ui, sans-serif',
                                size: 16,
                                weight: '600'
                            },
                            color: '#111827',
                            padding: {
                                bottom: 20
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Cooperation Probability (%)',
                                font: {
                                    family: 'Inter, system-ui, sans-serif',
                                    size: 12,
                                    weight: '500'
                                },
                                color: '#6B7280'
                            },
                            grid: {
                                display: true,
                                color: '#F3F4F6',
                                drawBorder: false
                            },
                            ticks: {
                                color: '#6B7280',
                                font: {
                                    family: 'Inter, system-ui, sans-serif',
                                    size: 10
                                },
                                maxTicksLimit: 11
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Total Coins Earned',
                                font: {
                                    family: 'Inter, system-ui, sans-serif',
                                    size: 12,
                                    weight: '500'
                                },
                                color: '#6B7280'
                            },
                            beginAtZero: true,
                            grid: {
                                display: true,
                                color: '#F3F4F6',
                                drawBorder: false
                            },
                            ticks: {
                                color: '#6B7280',
                                font: {
                                    family: 'Inter, system-ui, sans-serif',
                                    size: 10
                                }
                            }
                        }
                    },
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    animation: {
                        duration: 0 // Disable animation for real-time updates
                    }
                }
            });
        }
    </script>
    <script>
        // Create footer buttons and modals, and move explanations into them
        document.addEventListener('DOMContentLoaded', function () {
            try {
                const intro = document.querySelector('.introduction');
                if (!intro) return;

                const prisonersSection = intro.querySelector('.payoff-matrix');
                const monteCarloSection = intro.querySelector('.monte-carlo-explanation');

                // Build footer
                const footer = document.createElement('div');
                footer.className = 'explain-footer';

                const pButton = document.createElement('button');
                pButton.className = 'btn btn-outline';
                pButton.textContent = "What is the Prisoner's Dilemma?";
                pButton.onclick = () => openExplainModal('prisoners');

                const mButton = document.createElement('button');
                mButton.className = 'btn btn-outline';
                mButton.textContent = 'What is a Monte Carlo Simulation?';
                mButton.onclick = () => openExplainModal('montecarlo');

                footer.appendChild(pButton);
                footer.appendChild(mButton);
                document.body.appendChild(footer);

                // Build modals
                const overlay = document.createElement('div');
                overlay.className = 'explain-modal-overlay';
                overlay.id = 'explainOverlay';
                overlay.addEventListener('click', (e) => {
                    if (e.target === overlay) closeExplainModal();
                });

                const modal = document.createElement('div');
                modal.className = 'explain-modal';
                modal.setAttribute('role', 'dialog');
                modal.setAttribute('aria-modal', 'true');
                modal.id = 'explainModal';

                const header = document.createElement('div');
                header.className = 'explain-modal-header';
                const title = document.createElement('div');
                title.className = 'explain-modal-title';
                title.id = 'explainTitle';
                const closeBtn = document.createElement('button');
                closeBtn.className = 'explain-modal-close';
                closeBtn.textContent = 'Close';
                closeBtn.onclick = closeExplainModal;
                header.appendChild(title);
                header.appendChild(closeBtn);

                const body = document.createElement('div');
                body.className = 'explain-modal-body';
                body.id = 'explainBody';

                modal.appendChild(header);
                modal.appendChild(body);
                overlay.appendChild(modal);
                document.body.appendChild(overlay);

                // Move sections into off-DOM containers first to avoid duplication
                const stash = {
                    prisoners: prisonersSection ? prisonersSection : null,
                    montecarlo: monteCarloSection ? monteCarloSection : null
                };

                // If found, remove them from original spot and hide introduction if empty
                if (stash.prisoners && stash.prisoners.parentElement) {
                    stash.prisoners.parentElement.removeChild(stash.prisoners);
                }
                if (stash.montecarlo && stash.montecarlo.parentElement) {
                    stash.montecarlo.parentElement.removeChild(stash.montecarlo);
                }

                // Hide the intro container if it no longer has the two sections
                if (intro) {
                    intro.style.display = 'none';
                }

                // Store on window for open function
                window.__EXPLAIN_STASH__ = stash;
            } catch (e) {
                console.warn('Explain footer setup failed:', e);
            }
        });

        function openExplainModal(kind) {
            const overlay = document.getElementById('explainOverlay');
            const body = document.getElementById('explainBody');
            const title = document.getElementById('explainTitle');
            const stash = window.__EXPLAIN_STASH__ || {};

            // Clear previous content
            while (body.firstChild) body.removeChild(body.firstChild);

            if (kind === 'prisoners' && stash.prisoners) {
                title.textContent = "What is the Prisoner's Dilemma?";
                body.appendChild(stash.prisoners);
            } else if (kind === 'montecarlo' && stash.montecarlo) {
                title.textContent = 'What is a Monte Carlo Simulation?';
                body.appendChild(stash.montecarlo);
            } else {
                title.textContent = '';
            }

            overlay.style.display = 'flex';
            document.addEventListener('keydown', escToCloseHandler);
        }

        function closeExplainModal() {
            const overlay = document.getElementById('explainOverlay');
            overlay.style.display = 'none';
            document.removeEventListener('keydown', escToCloseHandler);
        }

        function escToCloseHandler(e) {
            if (e.key === 'Escape') closeExplainModal();
        }
    </script>
</body>
</html>
    """)

@app.route('/simulate', methods=['POST'])
def simulate():
    """Run a single Monte Carlo simulation with custom parameters"""
    try:
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
        
        data = request.json
        p1 = data.get("player1_prob", 0.5)
        p2 = data.get("player2_prob", 0.5)
        rounds = data.get("rounds", 1000)
        strategy1 = data.get("strategy1", "probabilistic")
        strategy2 = data.get("strategy2", "probabilistic")
        random_seed = data.get("random_seed", None)

        # Input validation
        if not (0 <= p1 <= 1) or not (0 <= p2 <= 1):
            return jsonify({"error": "Probabilities must be between 0 and 1"}), 400
        
        if rounds <= 0 or rounds > 1000000:
            return jsonify({"error": "Rounds must be between 1 and 1,000,000"}), 400

        # Run Monte Carlo simulation
        result = run_monte_carlo_simulation(p1, p2, rounds, strategy1, strategy2, random_seed)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"Simulation failed: {str(e)}"}), 500

@app.route('/start_simulation', methods=['POST'])
def start_simulation():
    """Start the parameter sweep simulation"""
    global simulation_thread, current_simulation
    
    if current_simulation and current_simulation.get('status') == 'running':
        return jsonify({"error": "Simulation already running"}), 400
    
    data = request.json
    rounds_per_config = data.get('rounds_per_config', 100)
    step_size = data.get('step_size', 0.01)
    
    def progress_callback(update_data):
        simulation_queue.put(update_data)
    
    # Start simulation in background thread
    simulation_thread = threading.Thread(
        target=run_parameter_sweep_background,
        args=(rounds_per_config, step_size, progress_callback)
    )
    simulation_thread.daemon = True
    simulation_thread.start()
    
    return jsonify({"message": "Simulation started"})

@app.route('/stop_simulation', methods=['POST'])
def stop_simulation():
    """Stop the current simulation"""
    global current_simulation
    
    if current_simulation:
        current_simulation['status'] = 'stopped'
    
    return jsonify({"message": "Simulation stopped"})

@app.route('/simulation_stream')
def simulation_stream():
    """Server-Sent Events stream for real-time updates"""
    def generate():
        while True:
            try:
                # Get update from queue (blocking with timeout)
                update_data = simulation_queue.get(timeout=1)
                yield f"data: {json.dumps(update_data)}\n\n"
                
                if update_data.get('type') == 'complete' or update_data.get('type') == 'error':
                    break
                    
            except queue.Empty:
                # Send keep-alive
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
                break
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/simulation_status')
def simulation_status():
    """Get current simulation status"""
    global current_simulation
    
    if not current_simulation:
        return jsonify({"status": "idle"})
    
    return jsonify(current_simulation)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)

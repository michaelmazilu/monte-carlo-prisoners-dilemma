from flask import Flask, request, jsonify, Response, render_template_string
from flask_cors import CORS
import torch
import numpy as np
import json
import time
import threading
import queue

app = Flask(__name__)
CORS(app)

# Globals for SSE single-run simulation
run_queue = queue.Queue()
run_thread = None
current_run = None

def run_monte_carlo_simulation(p1_defect, p2_defect, rounds):
    """Run Monte Carlo simulation using defection probabilities.

    p1_defect/p2_defect are probabilities in [0, 1] that each player defects on a round.
    We model cooperate as the complement of defect.
    """

    # Handle zero rounds safely
    if rounds == 0:
        return {
            "player1_avg_payoff": 0.0,
            "player2_avg_payoff": 0.0,
            "player1_coop_rate": 0.0,
            "player2_coop_rate": 0.0,
            "both_cooperate_rate": 0.0,
            "both_defect_rate": 0.0,
            "p1_coop_p2_defect_rate": 0.0,
            "p1_defect_p2_coop_rate": 0.0,
            "total_rounds": 0,
            "parameters": {
                "player1_defect_prob": p1_defect,
                "player2_defect_prob": p2_defect,
                "rounds": 0
            }
        }

    # Generate cooperate/defect actions
    p1_defects = torch.rand(rounds) < p1_defect
    p2_defects = torch.rand(rounds) < p2_defect
    p1_coops = ~p1_defects
    p2_coops = ~p2_defects

    # Payoff matrix calculation
    payoffs = torch.zeros((rounds, 2))

    payoffs[:, 0] = torch.where(
        p1_coops & p2_coops, 3,
        torch.where(p1_coops & ~p2_coops, 0,
                   torch.where(~p1_coops & p2_coops, 5, 1))
    )

    payoffs[:, 1] = torch.where(
        p1_coops & p2_coops, 3,
        torch.where(p1_coops & ~p2_coops, 5,
                   torch.where(~p1_coops & p2_coops, 0, 1))
    )

    # Calculate statistics
    result = {
        "player1_avg_payoff": payoffs[:, 0].float().mean().item(),
        "player2_avg_payoff": payoffs[:, 1].float().mean().item(),
        "player1_coop_rate": float(p1_coops.sum()) / rounds,
        "player2_coop_rate": float(p2_coops.sum()) / rounds,
        "both_cooperate_rate": float((p1_coops & p2_coops).sum()) / rounds,
        "both_defect_rate": float((~p1_coops & ~p2_coops).sum()) / rounds,
        "p1_coop_p2_defect_rate": float((p1_coops & ~p2_coops).sum()) / rounds,
        "p1_defect_p2_coop_rate": float((~p1_coops & p2_coops).sum()) / rounds,
        "total_rounds": rounds,
        "parameters": {
            "player1_defect_prob": p1_defect,
            "player2_defect_prob": p2_defect,
            "rounds": rounds
        }
    }

    return result


def run_simulation_streaming(p1_defect, p2_defect, rounds, batch_size, progress_callback):
    """Run simulation in batches using PyTorch and stream progress via callback."""
    global current_run
    try:
        current_run = {
            "status": "running",
            "rounds": rounds,
            "completed": 0,
            "start_time": time.time(),
        }

        if rounds == 0:
            # Immediate completion
            summary = run_monte_carlo_simulation(p1_defect, p2_defect, 0)
            progress_callback({"type": "complete", "result": summary})
            current_run["status"] = "completed"
            return

        # Accumulators
        p1_total = 0.0
        p2_total = 0.0
        both_coop = 0
        both_defect = 0
        p1_coop_p2_defect = 0
        p1_defect_p2_coop = 0

        remaining = rounds
        while remaining > 0 and current_run and current_run.get("status") == "running":
            b = min(batch_size, remaining)

            # Sample defects with PyTorch
            p1_def = torch.rand(b) < p1_defect
            p2_def = torch.rand(b) < p2_defect
            p1_coop = ~p1_def
            p2_coop = ~p2_def

            # Outcomes counts
            bc = (p1_coop & p2_coop).sum().item()
            bd = ((~p1_coop) & (~p2_coop)).sum().item()
            c_d = (p1_coop & (~p2_coop)).sum().item()
            d_c = ((~p1_coop) & p2_coop).sum().item()

            both_coop += int(bc)
            both_defect += int(bd)
            p1_coop_p2_defect += int(c_d)
            p1_defect_p2_coop += int(d_c)

            # Payoffs using vectorized torch.where
            payoffs_p1 = torch.where(
                p1_coop & p2_coop, torch.tensor(3.0),
                torch.where(p1_coop & (~p2_coop), torch.tensor(0.0),
                           torch.where((~p1_coop) & p2_coop, torch.tensor(5.0), torch.tensor(1.0)))
            )
            payoffs_p2 = torch.where(
                p1_coop & p2_coop, torch.tensor(3.0),
                torch.where(p1_coop & (~p2_coop), torch.tensor(5.0),
                           torch.where((~p1_coop) & p2_coop, torch.tensor(0.0), torch.tensor(1.0)))
            )

            p1_total += float(payoffs_p1.sum().item())
            p2_total += float(payoffs_p2.sum().item())

            current_run["completed"] += b
            remaining -= b

            # Progress payload
            completed = current_run["completed"]
            p1_avg = p1_total / completed
            p2_avg = p2_total / completed
            total_avg = (p1_total + p2_total) / completed
            progress = (completed / rounds) * 100.0

            progress_callback({
                "type": "progress",
                "completed": completed,
                "total": rounds,
                "progress": progress,
                "metrics": {
                    "p1_avg": p1_avg,
                    "p2_avg": p2_avg,
                    "total_avg": total_avg,
                    "both_cooperate": both_coop,
                    "both_defect": both_defect,
                    "p1_coop_p2_defect": p1_coop_p2_defect,
                    "p1_defect_p2_coop": p1_defect_p2_coop
                }
            })

        if current_run and current_run.get("status") == "running":
            # Finalize
            summary = {
                "player1_avg_payoff": p1_total / rounds,
                "player2_avg_payoff": p2_total / rounds,
                "player1_coop_rate": both_coop / rounds + p1_coop_p2_defect / rounds,
                "player2_coop_rate": both_coop / rounds + p1_defect_p2_coop / rounds,
                "both_cooperate_rate": both_coop / rounds,
                "both_defect_rate": both_defect / rounds,
                "p1_coop_p2_defect_rate": p1_coop_p2_defect / rounds,
                "p1_defect_p2_coop_rate": p1_defect_p2_coop / rounds,
                "total_rounds": rounds,
                "parameters": {
                    "player1_defect_prob": p1_defect,
                    "player2_defect_prob": p2_defect,
                    "rounds": rounds
                }
            }
            progress_callback({"type": "complete", "result": summary})
            current_run["status"] = "completed"
    except Exception as e:
        if current_run is not None:
            current_run["status"] = "error"
        progress_callback({"type": "error", "error": str(e)})

# Sweep functionality removed

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
            <h1>Monte Carlo Prisoner's Dilemma Simulator</h1>
            <p>Single-run simulation using defection probabilities</p>
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
            <h2>Simulation Parameters</h2>
            <p>Pick defection probabilities and number of rounds:</p>
            
            <div class="parameter-grid">
                <div class="parameter-group">
                    <label for="player1_prob">Player 1 Defection Probability</label>
                    <div class="slider-container">
                        <input type="range" id="player1_prob" min="0" max="100" value="50" class="slider">
                        <span id="player1_value">50%</span>
                    </div>
                </div>
                
                <div class="parameter-group">
                    <label for="player2_prob">Player 2 Defection Probability</label>
                    <div class="slider-container">
                        <input type="range" id="player2_prob" min="0" max="100" value="50" class="slider">
                        <span id="player2_value">50%</span>
                    </div>
                </div>
                
                <div class="parameter-group">
                    <label for="rounds">Number of Rounds</label>
                    <div class="slider-container">
                        <input type="range" id="rounds" min="0" max="50000" value="1000" step="100" class="slider">
                        <span id="rounds_value">1,000</span>
                    </div>
                </div>
            </div>
            
            <div class="button-group">
                <button class="btn btn-primary" id="runCustomBtn" onclick="runCustomSimulation()">Run Simulation</button>
            </div>
        </div>

        <div class="progress-container" id="progressContainer" style="display: none;">
            <h3>Simulation Progress</h3>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <div class="progress-text" id="progressText">Ready to start...</div>
            
            <div class="config-display" id="currentConfig" style="display: none;"></div>
        </div>

        <div class="stats-grid" id="statsGrid" style="display: none;"></div>

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
            <div class="chart-container" style="grid-column: 1 / -1;">
                <canvas id="coopScatterChart"></canvas>
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
            const chartsContainer = document.getElementById('chartsContainer');
            
            // Get parameter values
            const player1Prob = parseFloat(document.getElementById('player1_prob').value) / 100;
            const player2Prob = parseFloat(document.getElementById('player2_prob').value) / 100;
            const rounds = parseInt(document.getElementById('rounds').value);
            
            // Validate parameters
            if (rounds < 0 || rounds > 50000) {
                alert('Number of rounds must be between 0 and 50,000');
                return;
            }
            
            runCustomBtn.disabled = true;
            progressContainer.style.display = 'block';
            chartsContainer.style.display = 'grid';
            
            // Show loading state
            document.getElementById('progressText').textContent = 'Starting simulation...';
            document.getElementById('progressFill').style.width = '0%';

            // Create/Reset charts for live run (data will stream from SSE)
            if (charts.p1) charts.p1.destroy();
            if (charts.p2) charts.p2.destroy();
            if (charts.total) charts.total.destroy();
            if (charts.coop) charts.coop.destroy();

            charts.p1 = createLineChart('p1Chart', [], [], 'Player 1 Average Payoff (live)', '#6B7280');
            charts.p2 = createLineChart('p2Chart', [], [], 'Player 2 Average Payoff (live)', '#9CA3AF');
            charts.total = createScatterCombinedChart('totalChart');
            charts.coop = createPieChart('coopChart',
                ['Both Cooperate', 'Both Defect', 'P1 Coop, P2 Defect', 'P1 Defect, P2 Coop'],
                [0, 0, 0, 0],
                'Outcome Distribution (live)',
                ['#10B981', '#6B7280', '#F59E0B', '#EF4444']
            );
            charts.coopScatter = createCoopScatterChart('coopScatterChart');

            // SSE event handling
            const labels = [];
            const p1Series = [];
            const p2Series = [];
            const totalScatter = []; // {x,y} points of total average payoff
            const coopP1Scatter = []; // {x,y} y=cumulative coop rate of p1
            const coopP2Scatter = []; // {x,y} y=cumulative coop rate of p2

            if (eventSource) { try { eventSource.close(); } catch(e) {} }

            // Open SSE with parameters in the URL; backend will start run if needed
            const params = new URLSearchParams({
                p1: String(player1Prob),
                p2: String(player2Prob),
                rounds: String(rounds),
                batch: String(Math.max(10, Math.floor(rounds / 100)))
            });
            eventSource = new EventSource('/run_stream?' + params.toString());
            eventSource.onmessage = onSseMessage;
            eventSource.onerror = onSseError;

            function onSseMessage(event) {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'progress') {
                        const { completed, total, progress, metrics } = data;
                        document.getElementById('progressFill').style.width = progress + '%';
                        document.getElementById('progressText').textContent = `Processing ${Math.round(progress)}%... (${completed}/${total})`;

                        // Append series points
                        labels.push(completed);
                        p1Series.push(metrics.p1_avg);
                        p2Series.push(metrics.p2_avg);
                        totalSeries.push(metrics.total_avg);

                        charts.p1.data.labels = labels;
                        charts.p1.data.datasets[0].data = p1Series;
                        charts.p1.update('none');

                        charts.p2.data.labels = labels;
                        charts.p2.data.datasets[0].data = p2Series;
                        charts.p2.update('none');

                        // Update total scatter: point for total avg, lines for p1/p2
                        totalScatter.push({ x: completed, y: metrics.total_avg });
                        charts.total.data.datasets[0].data = totalScatter; // scatter points
                        charts.total.data.datasets[1].data = labels.map((x, i) => ({ x: labels[i], y: p1Series[i] }));
                        charts.total.data.datasets[2].data = labels.map((x, i) => ({ x: labels[i], y: p2Series[i] }));
                        charts.total.update('none');

                        charts.coop.data.datasets[0].data = [
                            metrics.both_cooperate,
                            metrics.both_defect,
                            metrics.p1_coop_p2_defect,
                            metrics.p1_defect_p2_coop
                        ];
                        charts.coop.update('none');

                        // Update cooperation scatter (cumulative rate per player)
                        const p1CoopCount = metrics.both_cooperate + metrics.p1_coop_p2_defect;
                        const p2CoopCount = metrics.both_cooperate + metrics.p1_defect_p2_coop;
                        coopP1Scatter.push({ x: completed, y: p1CoopCount / completed });
                        coopP2Scatter.push({ x: completed, y: p2CoopCount / completed });
                        charts.coopScatter.data.datasets[0].data = coopP1Scatter;
                        charts.coopScatter.data.datasets[1].data = coopP2Scatter;
                        charts.coopScatter.update('none');
                    } else if (data.type === 'complete') {
                        document.getElementById('progressFill').style.width = '100%';
                        document.getElementById('progressText').textContent = 'Simulation completed!';
                        runCustomBtn.disabled = false;
                        eventSource.close();
                        eventSource = null;
                    } else if (data.type === 'error') {
                        alert('Simulation error: ' + data.error);
                        runCustomBtn.disabled = false;
                        eventSource.close();
                        eventSource = null;
                    }
                } catch (e) {
                    console.error('SSE parse error', e);
                }
            }

            function onSseError() {
                console.error('SSE connection error');
                runCustomBtn.disabled = false;
                try { eventSource.close(); } catch(e) {}
                eventSource = null;
            }
        }
        // Simple chart data holder (used by custom charts only)
        let chartData = {};

        function displayCustomResults(result) {
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
                    labels: labels || [],
                    datasets: [{
                        data: data || [],
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

        function createScatterCombinedChart(canvasId) {
            const ctx = document.getElementById(canvasId).getContext('2d');
            return new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [
                        {
                            type: 'scatter',
                            label: 'Total Avg Payoff (scatter)',
                            data: [],
                            backgroundColor: '#111827',
                            borderColor: '#111827',
                            pointRadius: 3
                        },
                        {
                            type: 'line',
                            label: 'P1 Avg Payoff',
                            data: [], // as {x,y}
                            borderColor: '#ef4444',
                            backgroundColor: '#ef4444',
                            fill: false,
                            tension: 0,
                            pointRadius: 0,
                            borderWidth: 2
                        },
                        {
                            type: 'line',
                            label: 'P2 Avg Payoff',
                            data: [], // as {x,y}
                            borderColor: '#3b82f6',
                            backgroundColor: '#3b82f6',
                            fill: false,
                            tension: 0,
                            pointRadius: 0,
                            borderWidth: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: true, position: 'top' },
                        title: {
                            display: true,
                            text: 'Total Average Payoff (scatter) + P1/P2 lines',
                            font: { family: 'Inter, system-ui, sans-serif', size: 14, weight: '500' },
                            color: '#374151'
                        }
                    },
                    parsing: false,
                    scales: {
                        x: {
                            type: 'linear',
                            title: { display: true, text: 'Round', color: '#6B7280' },
                            grid: { display: true, color: '#F3F4F6' },
                            ticks: { color: '#6B7280' }
                        },
                        y: {
                            title: { display: true, text: 'Coins', color: '#6B7280' },
                            grid: { display: true, color: '#F3F4F6' },
                            ticks: { color: '#6B7280' }
                        }
                    }
                }
            });
        }

        function createCoopScatterChart(canvasId) {
            const ctx = document.getElementById(canvasId).getContext('2d');
            return new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [
                        {
                            label: 'P1 Cooperation Rate',
                            data: [],
                            backgroundColor: '#ef4444',
                            borderColor: '#ef4444',
                            pointRadius: 2,
                            showLine: true,
                            borderWidth: 2
                        },
                        {
                            label: 'P2 Cooperation Rate',
                            data: [],
                            backgroundColor: '#3b82f6',
                            borderColor: '#3b82f6',
                            pointRadius: 2,
                            showLine: true,
                            borderWidth: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: true, position: 'top' },
                        title: {
                            display: true,
                            text: 'Cooperation Rate over Rounds (scatter)',
                            font: { family: 'Inter, system-ui, sans-serif', size: 14, weight: '500' },
                            color: '#374151'
                        }
                    },
                    parsing: false,
                    scales: {
                        x: {
                            type: 'linear',
                            title: { display: true, text: 'Round', color: '#6B7280' },
                            grid: { display: true, color: '#F3F4F6' },
                            ticks: { color: '#6B7280' }
                        },
                        y: {
                            min: 0,
                            max: 1,
                            title: { display: true, text: 'Cooperation Rate', color: '#6B7280' },
                            grid: { display: true, color: '#F3F4F6' },
                            ticks: { color: '#6B7280' }
                        }
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
        p1_defect = data.get("player1_defect_prob", 0.5)
        p2_defect = data.get("player2_defect_prob", 0.5)
        rounds = data.get("rounds", 1000)

        # Input validation
        if not (0 <= p1_defect <= 1) or not (0 <= p2_defect <= 1):
            return jsonify({"error": "Probabilities must be between 0 and 1"}), 400
        
        if rounds < 0 or rounds > 50000:
            return jsonify({"error": "Rounds must be between 0 and 50,000"}), 400

        # Run Monte Carlo simulation
        result = run_monte_carlo_simulation(p1_defect, p2_defect, rounds)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"Simulation failed: {str(e)}"}), 500

@app.route('/start_run', methods=['POST'])
def start_run():
    """Start a single-run simulation in the background and stream progress via SSE."""
    global run_thread, current_run

    if current_run and current_run.get('status') == 'running':
        return jsonify({"error": "A simulation is already running"}), 400

    data = request.json or {}
    p1_defect = float(data.get('player1_defect_prob', 0.5))
    p2_defect = float(data.get('player2_defect_prob', 0.5))
    rounds = int(data.get('rounds', 1000))
    batch_size = int(data.get('batch_size', max(10, rounds // 100)))

    if not (0.0 <= p1_defect <= 1.0 and 0.0 <= p2_defect <= 1.0):
        return jsonify({"error": "Probabilities must be between 0 and 1"}), 400
    if rounds < 0 or rounds > 50000:
        return jsonify({"error": "Rounds must be between 0 and 50,000"}), 400

    # Clear any stale messages from previous runs to avoid instant completion
    try:
        while True:
            run_queue.get_nowait()
    except queue.Empty:
        pass

    def progress_callback(payload):
        run_queue.put(payload)

    # Mark current run as starting before thread begins
    current_run = {"status": "running", "rounds": rounds, "completed": 0, "start_time": time.time()}

    run_thread = threading.Thread(
        target=run_simulation_streaming,
        args=(p1_defect, p2_defect, rounds, batch_size, progress_callback)
    )
    run_thread.daemon = True
    run_thread.start()

    return jsonify({"message": "Simulation started"})

@app.route('/run_stream')
def run_stream():
    """SSE stream for the single-run simulation. If not running, start it using query params."""
    global run_thread, current_run

    # If a run is not active, start one from query params to avoid client -> server race
    if not (current_run and current_run.get('status') == 'running'):
        try:
            while True:
                run_queue.get_nowait()
        except queue.Empty:
            pass

        p1_defect = float(request.args.get('p1', 0.5))
        p2_defect = float(request.args.get('p2', 0.5))
        rounds = int(request.args.get('rounds', 1000))
        batch_size = int(request.args.get('batch', max(10, rounds // 100)))

        def progress_callback(payload):
            run_queue.put(payload)

        current_run = {"status": "running", "rounds": rounds, "completed": 0, "start_time": time.time()}
        run_thread = threading.Thread(
            target=run_simulation_streaming,
            args=(p1_defect, p2_defect, rounds, batch_size, progress_callback)
        )
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

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)

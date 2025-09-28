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
    <title>🎯 Monte Carlo Prisoner's Dilemma Simulator</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: #FFFFFF;
            color: #374151;
            line-height: 1.6;
            min-height: 100vh;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }

        .header {
            text-align: left;
            margin-bottom: 60px;
        }

        .header h1 {
            font-size: 2.25rem;
            font-weight: 600;
            color: #111827;
            margin-bottom: 8px;
        }

        .header p {
            font-size: 1.125rem;
            font-weight: 400;
            color: #6B7280;
        }

        .simulation-controls {
            background: #F9FAFB;
            border: 1px solid #E5E7EB;
            padding: 40px;
            margin-bottom: 40px;
            text-align: left;
        }

        .simulation-controls h2 {
            font-size: 1.5rem;
            font-weight: 600;
            color: #111827;
            margin-bottom: 16px;
        }

        .simulation-controls p {
            font-size: 1rem;
            color: #6B7280;
            margin-bottom: 12px;
        }

        .simulation-controls ul {
            margin: 20px 0 30px 0;
            padding-left: 20px;
        }

        .simulation-controls li {
            font-size: 0.875rem;
            color: #6B7280;
            margin-bottom: 4px;
        }

        .btn {
            font-family: 'Inter', system-ui, sans-serif;
            font-size: 0.875rem;
            font-weight: 500;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            margin-right: 12px;
            margin-bottom: 12px;
        }

        .btn-primary {
            background: #111827;
            color: white;
            border: 1px solid #111827;
        }

        .btn-primary:hover:not(:disabled) {
            background: #374151;
            border-color: #374151;
        }

        .btn-outline {
            background: transparent;
            color: #6B7280;
            border: 1px solid #D1D5DB;
        }

        .btn-outline:hover:not(:disabled) {
            background: #F9FAFB;
            color: #374151;
            border-color: #9CA3AF;
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .progress-container {
            background: #F9FAFB;
            border: 1px solid #E5E7EB;
            padding: 32px;
            margin: 40px 0;
            display: none;
        }

        .progress-container h3 {
            font-size: 1.25rem;
            font-weight: 600;
            color: #111827;
            margin-bottom: 24px;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: #E5E7EB;
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 16px;
        }

        .progress-fill {
            height: 100%;
            background: #6B7280;
            width: 0%;
            transition: width 0.3s ease;
        }

        .progress-text {
            font-size: 0.875rem;
            font-weight: 500;
            color: #374151;
            margin-bottom: 20px;
        }

        .config-display {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            padding: 16px;
            border-radius: 6px;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
            font-size: 0.75rem;
            color: #6B7280;
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
            color: #6B7280;
        }

        .stat-value {
            font-size: 1.125rem;
            font-weight: 600;
            color: #111827;
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
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            padding: 24px;
            border-radius: 8px;
        }

        .chart-container canvas {
            height: 300px !important;
        }

        .live-updates {
            background: #F9FAFB;
            border: 1px solid #E5E7EB;
            padding: 24px;
            margin: 40px 0;
            max-height: 240px;
            overflow-y: auto;
            border-radius: 8px;
        }

        .live-updates h4 {
            font-size: 1rem;
            font-weight: 600;
            color: #111827;
            margin-bottom: 16px;
        }

        .update-item {
            padding: 8px 0;
            border-bottom: 1px solid #E5E7EB;
            font-size: 0.75rem;
            color: #6B7280;
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Monte Carlo Prisoner's Dilemma Simulator</h1>
            <p>Real-time parameter sweep with 10,000 experiments</p>
        </div>

        <div class="simulation-controls">
            <h2>Start Your Experiment</h2>
            <p>Run a comprehensive parameter sweep:</p>
            <ul>
                <li><strong>Player 1:</strong> 100% → 0% cooperation (1% steps)</li>
                <li><strong>Player 2:</strong> 0% → 100% cooperation (1% steps)</li>
                <li><strong>Total:</strong> 10,000 experiments with 100 rounds each</li>
            </ul>
            <button class="btn btn-primary" id="startBtn" onclick="startSimulation()">Start Simulation</button>
            <button class="btn btn-outline" id="stopBtn" onclick="stopSimulation()" disabled>Stop</button>
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

        function startSimulation() {
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const progressContainer = document.getElementById('progressContainer');
            const statsGrid = document.getElementById('statsGrid');
            const chartsContainer = document.getElementById('chartsContainer');
            const histogramContainer = document.getElementById('histogramContainer');
            
            startBtn.disabled = true;
            stopBtn.disabled = false;
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
</body>
</html>
    """)

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

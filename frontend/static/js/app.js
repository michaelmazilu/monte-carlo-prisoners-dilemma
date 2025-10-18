const form = document.getElementById("simulation-form");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const statusText = document.getElementById("statusText");
const summaryContent = document.getElementById("summaryContent");

const strategySelects = {
    player1: document.getElementById("player1Strategy"),
    player2: document.getElementById("player2Strategy"),
};

const probabilityInputs = {
    player1: document.querySelector('[data-probability-for="player1"]'),
    player2: document.querySelector('[data-probability-for="player2"]'),
};

let eventSource = null;
let charts = null;

document.addEventListener("DOMContentLoaded", () => {
    charts = initialiseCharts();
    updateProbabilityInputs();
});

Object.values(strategySelects).forEach((select) => {
    select.addEventListener("change", updateProbabilityInputs);
});

stopButton.addEventListener("click", () => {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    startButton.disabled = false;
    stopButton.disabled = true;
    setStatus("Simulation stopped.", "warning");
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }

    resetCharts();
    summaryContent.innerHTML = "<p>Simulation running… waiting for summary.</p>";
    setStatus("Preparing simulation…", "info");
    startButton.disabled = true;
    stopButton.disabled = false;

    try {
        const config = buildSimulationConfig();
        const response = await fetch("/api/simulations", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(config),
        });

        if (!response.ok) {
            const errorPayload = await response.json().catch(() => ({}));
            throw new Error(errorPayload.error || "Failed to create simulation.");
        }

        const { simulation_id: simulationId } = await response.json();
        startEventStream(simulationId);
    } catch (error) {
        console.error(error);
        setStatus(error.message || "Unable to start simulation.", "danger");
        startButton.disabled = false;
        stopButton.disabled = true;
        summaryContent.innerHTML = "<p class=\"error\">Simulation failed to start.</p>";
    }
});

function buildSimulationConfig() {
    const rounds = Number.parseInt(form.rounds.value, 10);
    const monteCarloRuns = Number.parseInt(form.monteCarloRuns.value, 10);

    const strategies = ["player1", "player2"].map((playerKey) => {
        const type = strategySelects[playerKey].value;
        const result = { type };
        if (type === "probabilistic") {
            const input = probabilityInputs[playerKey].querySelector("input");
            result.cooperate_probability = Number.parseFloat(input.value);
        }
        return result;
    });

    return {
        rounds,
        monte_carlo_runs: monteCarloRuns,
        strategies,
    };
}

function startEventStream(simulationId) {
    if (eventSource) {
        eventSource.close();
    }

    eventSource = new EventSource(`/api/simulations/${simulationId}/stream`);
    setStatus("Simulation running…", "info");

    eventSource.addEventListener("round", (event) => {
        const payload = JSON.parse(event.data);
        handleRoundEvent(payload);
    });

    eventSource.addEventListener("run_complete", (event) => {
        const payload = JSON.parse(event.data);
        setStatus(`Run ${payload.run} complete.`, "success");
    });

    eventSource.addEventListener("summary", (event) => {
        const payload = JSON.parse(event.data);
        handleSummaryEvent(payload);
        setStatus("Simulation finished!", "success");
        startButton.disabled = false;
        stopButton.disabled = true;
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
    });

    eventSource.onerror = () => {
        setStatus("Connection lost. Please try again.", "danger");
        startButton.disabled = false;
        stopButton.disabled = true;
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
    };
}

function handleRoundEvent(payload) {
    const label = `Run ${payload.run} · Round ${payload.round}`;
    charts.totalCoins.data.labels.push(label);
    charts.totalCoins.data.datasets[0].data.push(payload.total_payoff.player1);
    charts.totalCoins.data.datasets[1].data.push(payload.total_payoff.player2);
    charts.totalCoins.update("none");

    charts.roundPayoff.data.labels.push(label);
    charts.roundPayoff.data.datasets[0].data.push(payload.round_payoff.player1);
    charts.roundPayoff.data.datasets[1].data.push(payload.round_payoff.player2);
    charts.roundPayoff.update("none");

    charts.cooperationRate.data.labels.push(label);
    charts.cooperationRate.data.datasets[0].data.push(
        Math.round(payload.cooperation_rate.player1 * 1000) / 10
    );
    charts.cooperationRate.data.datasets[1].data.push(
        Math.round(payload.cooperation_rate.player2 * 1000) / 10
    );
    charts.cooperationRate.update("none");

    const outcomeCounts = [
        payload.outcome_counts.CC,
        payload.outcome_counts.CD,
        payload.outcome_counts.DC,
        payload.outcome_counts.DD,
    ];
    charts.outcomeDistribution.data.datasets[0].data = outcomeCounts;
    charts.outcomeDistribution.update("none");
}

function handleSummaryEvent(payload) {
    const totalCoins = payload.total_payoff;
    const avgPayoff = payload.average_payoff_per_round;
    const cooperationRates = payload.cooperation_rate;

    summaryContent.innerHTML = `
        <div class="metric">
            <span><strong>Total Coins</strong><br>Player 1: ${totalCoins.player1.toFixed(
                2
            )} | Player 2: ${totalCoins.player2.toFixed(2)}</span>
            <span><strong>Average Payoff / Round</strong><br>Player 1: ${avgPayoff.player1.toFixed(
                3
            )} | Player 2: ${avgPayoff.player2.toFixed(3)}</span>
        </div>
        <div class="metric">
            <span><strong>Cooperation Rate</strong><br>Player 1: ${(cooperationRates.player1 * 100).toFixed(
                1
            )}% | Player 2: ${(cooperationRates.player2 * 100).toFixed(1)}%</span>
            <span><strong>Runs × Rounds</strong><br>${payload.runs} runs · ${payload.rounds} rounds each</span>
        </div>
        <p><strong>Outcome Counts:</strong> CC ${payload.outcome_counts.CC}, CD ${
        payload.outcome_counts.CD
    }, DC ${payload.outcome_counts.DC}, DD ${payload.outcome_counts.DD}</p>
        <p><strong>Outcome Distribution:</strong> CC ${(payload.outcome_distribution.CC * 100).toFixed(
            1
        )}%, CD ${(payload.outcome_distribution.CD * 100).toFixed(1)}%, DC ${(
        payload.outcome_distribution.DC * 100
    ).toFixed(1)}%, DD ${(payload.outcome_distribution.DD * 100).toFixed(1)}%</p>
    `;
}

function resetCharts() {
    Object.values(charts).forEach((chart) => {
        chart.data.labels.length = 0;
        chart.data.datasets.forEach((dataset) => {
            dataset.data.length = 0;
        });
        chart.update("none");
    });
}

function initialiseCharts() {
    const totalCoinsChart = new Chart(document.getElementById("totalCoinsChart"), {
        type: "line",
        data: {
            labels: [],
            datasets: [
                datasetConfig("Player 1", "#38bdf8"),
                datasetConfig("Player 2", "#f472b6"),
            ],
        },
        options: baseLineChartOptions("Coins"),
    });

    const roundPayoffChart = new Chart(document.getElementById("roundPayoffChart"), {
        type: "bar",
        data: {
            labels: [],
            datasets: [
                datasetConfig("Player 1", "#22c55e"),
                datasetConfig("Player 2", "#f97316"),
            ],
        },
        options: {
            ...baseBarChartOptions("Payoff"),
            scales: {
                x: { stacked: true },
                y: { beginAtZero: true },
            },
        },
    });

    const cooperationRateChart = new Chart(
        document.getElementById("cooperationRateChart"),
        {
            type: "line",
            data: {
                labels: [],
                datasets: [
                    datasetConfig("Player 1", "#60a5fa"),
                    datasetConfig("Player 2", "#facc15"),
                ],
            },
            options: {
                ...baseLineChartOptions("Cooperation %"),
                scales: {
                    y: {
                        beginAtZero: true,
                        suggestedMax: 100,
                        ticks: {
                            callback: (value) => `${value}%`,
                        },
                    },
                },
            },
        }
    );

    const outcomeDistributionChart = new Chart(
        document.getElementById("outcomeDistributionChart"),
        {
            type: "doughnut",
            data: {
                labels: ["CC", "CD", "DC", "DD"],
                datasets: [
                    {
                        label: "Outcomes",
                        data: [0, 0, 0, 0],
                        backgroundColor: ["#38bdf8", "#22c55e", "#f97316", "#f43f5e"],
                        borderWidth: 0,
                    },
                ],
            },
            options: {
                plugins: {
                    legend: { position: "bottom", labels: { color: "#cbd5f5" } },
                },
            },
        }
    );

    return {
        totalCoins: totalCoinsChart,
        roundPayoff: roundPayoffChart,
        cooperationRate: cooperationRateChart,
        outcomeDistribution: outcomeDistributionChart,
    };
}

function datasetConfig(label, color) {
    return {
        label,
        data: [],
        borderColor: color,
        backgroundColor: color,
        fill: false,
        tension: 0.25,
        pointRadius: 0,
        borderWidth: 2,
    };
}

function baseLineChartOptions(suffix) {
    return {
        animation: false,
        responsive: true,
        scales: {
            x: {
                ticks: { color: "#cbd5f5" },
                grid: { color: "rgba(148, 163, 184, 0.1)" },
            },
            y: {
                beginAtZero: true,
                ticks: {
                    color: "#cbd5f5",
                    callback: (value) => `${value} ${suffix}`,
                },
                grid: { color: "rgba(148, 163, 184, 0.12)" },
            },
        },
        plugins: {
            legend: {
                position: "bottom",
                labels: { color: "#cbd5f5" },
            },
        },
    };
}

function baseBarChartOptions(suffix) {
    return {
        animation: false,
        responsive: true,
        plugins: {
            legend: {
                position: "bottom",
                labels: { color: "#cbd5f5" },
            },
        },
        scales: {
            x: {
                ticks: { color: "#cbd5f5" },
                grid: { color: "rgba(148, 163, 184, 0.1)" },
            },
            y: {
                beginAtZero: true,
                ticks: {
                    color: "#cbd5f5",
                    callback: (value) => `${value} ${suffix}`,
                },
                grid: { color: "rgba(148, 163, 184, 0.12)" },
            },
        },
    };
}

function updateProbabilityInputs() {
    ["player1", "player2"].forEach((playerKey) => {
        const select = strategySelects[playerKey];
        const wrapper = probabilityInputs[playerKey];
        if (select.value === "probabilistic") {
            wrapper.classList.remove("hidden");
        } else {
            wrapper.classList.add("hidden");
        }
    });
}

function setStatus(message, level = "info") {
    statusText.textContent = message;
    statusText.dataset.level = level;
}

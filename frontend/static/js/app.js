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

const PLAYER_KEYS = ["player1", "player2"];
const PLAYER_COLORS = {
    player1: "#38bdf8",
    player2: "#f97316",
};

let eventSource = null;
let charts = null;
const lifecycle = {
    isRunning: false,
    isCompleted: false,
    isStopped: false,
};

document.addEventListener("DOMContentLoaded", () => {
    charts = initialiseCharts();
    updateProbabilityInputs();
    resetPlayerStats();
});

Object.values(strategySelects).forEach((select) => {
    select.addEventListener("change", updateProbabilityInputs);
});

stopButton.addEventListener("click", () => {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    lifecycle.isRunning = false;
    lifecycle.isStopped = true;
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
    lifecycle.isRunning = true;
    lifecycle.isCompleted = false;
    lifecycle.isStopped = false;

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
        lifecycle.isRunning = false;
        summaryContent.innerHTML = "<p class=\"error\">Simulation failed to start.</p>";
    }
});

function buildSimulationConfig() {
    const rounds = Number.parseInt(form.rounds.value, 10);
    const monteCarloRuns = Number.parseInt(form.monteCarloRuns.value, 10);

    const strategies = PLAYER_KEYS.map((playerKey) => {
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
        lifecycle.isRunning = false;
        lifecycle.isCompleted = true;
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
    });

    eventSource.onerror = (error) => {
        const currentSource = eventSource;
        if (!lifecycle.isRunning || lifecycle.isCompleted || lifecycle.isStopped) {
            if (currentSource) {
                currentSource.close();
                eventSource = null;
            }
            return;
        }
        console.error("SSE connection error", error);
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
    PLAYER_KEYS.forEach((playerKey) => {
        try {
            const playerCharts = charts[playerKey];
            if (!playerCharts) {
                return;
            }

            const totals = payload.total_payoff ?? {};
            const totalCoins = totals[playerKey];
            if (typeof totalCoins !== "number") {
                console.warn(`Missing total coins for ${playerKey}`, payload);
                return;
            }

            const cooperated = payload.cooperated?.[playerKey] ?? false;

            playerCharts.coins.data.labels.push(label);
            playerCharts.coins.data.datasets[0].data.push(totalCoins);
            playerCharts.coins.update("none");

            playerCharts.cooperation.data.labels.push(label);
            playerCharts.cooperation.data.datasets[0].data.push(
                cooperated ? 1 : 0
            );
            playerCharts.cooperation.update("none");

            updatePlayerStatsDuringRun(playerKey, payload, totalCoins);
        } catch (error) {
            console.error(`Failed to update data for ${playerKey}`, error, payload);
        }
    });
}

function handleSummaryEvent(payload) {
    PLAYER_KEYS.forEach((playerKey) => {
        const totals = payload.total_payoff ?? {};
        const averages = payload.average_payoff_per_round ?? {};
        const rates = payload.cooperation_rate ?? {};
        const coopTotals = payload.total_cooperation ?? {};

        const totalCoins = totals[playerKey];
        const averagePayoff = averages[playerKey];
        const cooperationRate = rates[playerKey];
        const totalCooperation = coopTotals[playerKey];

        if (
            typeof totalCoins !== "number" ||
            typeof averagePayoff !== "number" ||
            typeof cooperationRate !== "number" ||
            typeof totalCooperation !== "number"
        ) {
            return;
        }

        setPlayerStat(
            playerKey,
            "totalCoins",
            totalCoins.toFixed(2)
        );
        setPlayerStat(
            playerKey,
            "avgPayoff",
            averagePayoff.toFixed(3)
        );
        setPlayerStat(
            playerKey,
            "coopRate",
            `${(cooperationRate * 100).toFixed(1)}%`
        );
        setPlayerStat(
            playerKey,
            "totalCooperation",
            totalCooperation.toString()
        );
    });

    summaryContent.innerHTML = `
        <div class="metric">
            <span><strong>Runs</strong><br>${payload.runs}</span>
            <span><strong>Rounds / Run</strong><br>${payload.rounds}</span>
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
    if (!charts) {
        return;
    }

    Object.values(charts).forEach((playerCharts) => {
        playerCharts.coins.data.labels.length = 0;
        playerCharts.coins.data.datasets[0].data.length = 0;
        playerCharts.coins.update("none");

        playerCharts.cooperation.data.labels.length = 0;
        playerCharts.cooperation.data.datasets[0].data.length = 0;
        playerCharts.cooperation.update("none");
    });

    resetPlayerStats();
}

function initialiseCharts() {
    return PLAYER_KEYS.reduce((acc, playerKey) => {
        acc[playerKey] = {
            coins: createCoinsChart(
                document.getElementById(`${playerKey}CoinsChart`),
                PLAYER_COLORS[playerKey]
            ),
            cooperation: createCooperationChart(
                document.getElementById(`${playerKey}CooperationChart`),
                PLAYER_COLORS[playerKey]
            ),
        };
        return acc;
    }, {});
}

function createCoinsChart(context, color) {
    return new Chart(context, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Total Coins",
                    data: [],
                    borderColor: color,
                    backgroundColor: color,
                    tension: 0.25,
                    fill: false,
                    pointRadius: 0,
                    borderWidth: 2,
                },
            ],
        },
        options: {
            animation: false,
            responsive: true,
            plugins: {
                legend: { display: false },
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
                        callback: (value) => `${value} coins`,
                    },
                    grid: { color: "rgba(148, 163, 184, 0.12)" },
                },
            },
        },
    });
}

function createCooperationChart(context, color) {
    return new Chart(context, {
        type: "bar",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Cooperation",
                    data: [],
                    backgroundColor: color,
                    borderRadius: 6,
                },
            ],
        },
        options: {
            animation: false,
            responsive: true,
            plugins: {
                legend: { display: false },
            },
            scales: {
                x: {
                    ticks: { color: "#cbd5f5" },
                    grid: { display: false },
                },
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        stepSize: 1,
                        color: "#cbd5f5",
                        callback: (value) => (value === 1 ? "Cooperate" : "Defect"),
                    },
                    grid: { color: "rgba(148, 163, 184, 0.12)" },
                },
            },
        },
    });
}

function updatePlayerStatsDuringRun(playerKey, payload, totalCoins) {
    const roundsPlayed = payload.round;
    const cooperationTotals = payload.cumulative_cooperation ?? {};
    const cooperationRates = payload.cooperation_rate ?? {};
    const totalCooperation = cooperationTotals[playerKey];
    const cooperationRate = cooperationRates[playerKey];

    if (
        typeof totalCoins !== "number" ||
        typeof totalCooperation !== "number" ||
        typeof cooperationRate !== "number"
    ) {
        return;
    }

    const averagePayoff = roundsPlayed > 0 ? totalCoins / roundsPlayed : 0;

    setPlayerStat(playerKey, "totalCoins", totalCoins.toFixed(2));
    setPlayerStat(playerKey, "avgPayoff", averagePayoff.toFixed(3));
    setPlayerStat(
        playerKey,
        "coopRate",
        `${(cooperationRate * 100).toFixed(1)}%`
    );
    setPlayerStat(playerKey, "totalCooperation", totalCooperation.toString());
}

function resetPlayerStats() {
    PLAYER_KEYS.forEach((playerKey) => {
        ["totalCoins", "avgPayoff", "coopRate", "totalCooperation"].forEach(
            (field) => {
                setPlayerStat(playerKey, field, "--");
            }
        );
    });
}

function setPlayerStat(playerKey, field, value) {
    const element = document.querySelector(
        `.stat-value[data-player="${playerKey}"][data-field="${field}"]`
    );
    if (element) {
        element.textContent = value ?? "--";
    }
}

function updateProbabilityInputs() {
    PLAYER_KEYS.forEach((playerKey) => {
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

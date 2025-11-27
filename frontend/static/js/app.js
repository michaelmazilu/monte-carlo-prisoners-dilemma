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

const payoffInputs = {
    reward: document.getElementById("payoffReward"),
    temptation: document.getElementById("payoffTemptation"),
    sucker: document.getElementById("payoffSucker"),
    punishment: document.getElementById("payoffPunishment"),
};

const DEFAULT_PAYOFFS = {
    reward: 3,
    temptation: 5,
    sucker: 0,
    punishment: 1,
};

const CHART_UPDATE_INTERVAL = 25;
const MAX_CHART_POINTS = Infinity;

let payoffState = { ...DEFAULT_PAYOFFS };

const PLAYER_KEYS = ["player1", "player2"];
const PLAYER_COLORS = {
    player1: "#38bdf8",
    player2: "#f97316",
};

const pendingChartUpdates = PLAYER_KEYS.reduce((acc, key) => {
    acc[key] = false;
    return acc;
}, {});

let eventSource = null;
let charts = null;
let coinsAxisMax = 0;
const lifecycle = {
    isRunning: false,
    isCompleted: false,
    isStopped: false,
};

document.addEventListener("DOMContentLoaded", () => {
    charts = initialiseCharts();
    synchroniseCoinsAxes();
    updateProbabilityInputs();
    resetPlayerStats();
    setSummaryPlaceholder();
    initialisePayoffControls();
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
    setSummaryPlaceholder("Simulation running…");
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
        setSummaryPlaceholder("Simulation failed to start.");
    }
});

function buildSimulationConfig() {
    const rounds = Number.parseInt(form.rounds.value, 10);

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
        monte_carlo_runs: 1,
        strategies,
        payoffs: getPayoffValues(),
    };
}

function getPayoffValues() {
    return { ...payoffState };
}

function initialisePayoffControls() {
    payoffState = { ...DEFAULT_PAYOFFS };
    Object.entries(payoffInputs).forEach(([key, input]) => {
        if (!input) {
            return;
        }
        const parsed = Number.parseFloat(input.value);
        if (Number.isFinite(parsed)) {
            payoffState[key] = parsed;
        } else {
            input.value = formatInputValue(payoffState[key]);
        }
        input.setCustomValidity("");
    });
    attachPayoffInputListeners();
}

function attachPayoffInputListeners() {
    Object.entries(payoffInputs).forEach(([key, input]) => {
        if (!input) {
            return;
        }
        input.addEventListener("input", () => handlePayoffInputChange(key, input, false));
        input.addEventListener("change", () => handlePayoffInputChange(key, input, true));
        input.addEventListener("blur", () => handlePayoffInputChange(key, input, true));
    });
}

function handlePayoffInputChange(key, input, enforceValidity) {
    const parsed = Number.parseFloat(input.value);
    if (Number.isFinite(parsed)) {
        payoffState[key] = parsed;
        return;
    }

    if (!enforceValidity) {
        return;
    }

    const fallback = getPayoffFallbackValue(key);
    payoffState[key] = fallback;
    syncPayoffInputsFromState();
}

function getPayoffFallbackValue(key) {
    const current = payoffState[key];
    if (typeof current === "number" && Number.isFinite(current)) {
        return current;
    }
    return DEFAULT_PAYOFFS[key];
}

function syncPayoffInputsFromState() {
    Object.entries(payoffInputs).forEach(([key, input]) => {
        if (input) {
            input.value = formatInputValue(payoffState[key]);
        }
    });
}

function formatInputValue(value) {
    if (typeof value !== "number" || Number.isNaN(value)) {
        return "";
    }
    return Number.isInteger(value) ? value.toString() : String(value);
}

function startEventStream(simulationId) {
    if (eventSource) {
        eventSource.close();
    }

    eventSource = new EventSource(`/api/simulations/${simulationId}/stream`);
    setStatus("Simulation running…", "info");

    eventSource.addEventListener("round_batch", (event) => {
        const payload = JSON.parse(event.data);
        handleRoundBatchEvent(payload);
    });

    eventSource.addEventListener("round", (event) => {
        const payload = JSON.parse(event.data);
        handleRoundEvent(payload);
    });

    eventSource.addEventListener("run_complete", () => {
        if (!lifecycle.isCompleted) {
            setStatus("Simulation in progress…", "info");
        }
        flushPendingChartUpdates();
    });

    eventSource.addEventListener("summary", (event) => {
        const payload = JSON.parse(event.data);
        handleSummaryEvent(payload);
        flushPendingChartUpdates(true);
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
        setSummaryPlaceholder("Connection lost.");
    };
}

function handleRoundBatchEvent(payload) {
    const rounds = Array.isArray(payload.rounds) ? payload.rounds : [];
    rounds.forEach((roundPayload) => {
        handleRoundEvent(roundPayload);
    });
}

function handleRoundEvent(payload) {
    const label = `Round ${payload.round}`;
    const totals = payload.total_payoff ?? {};
    const maxCoinsThisRound = PLAYER_KEYS.reduce((acc, key) => {
        const value = totals[key];
        if (typeof value === "number") {
            return Math.max(acc, value);
        }
        return acc;
    }, 0);

    if (maxCoinsThisRound > coinsAxisMax) {
        coinsAxisMax = maxCoinsThisRound;
        synchroniseCoinsAxes();
    }

    PLAYER_KEYS.forEach((playerKey) => {
        try {
            const playerCharts = charts[playerKey];
            if (!playerCharts) {
                return;
            }

            const cooperationRates = payload.cooperation_rate ?? {};
            const totalCoins = totals[playerKey];
            const cooperationRate = cooperationRates[playerKey];
            if (
                typeof totalCoins !== "number" ||
                typeof cooperationRate !== "number"
            ) {
                console.warn(`Missing totals for ${playerKey}`, payload);
                return;
            }

            appendChartPoint(playerCharts.coins, label, totalCoins);
            appendChartPoint(
                playerCharts.cooperation,
                label,
                cooperationRate * 100
            );

            if (shouldUpdateChart(payload.round)) {
                updatePlayerCharts(playerCharts);
                pendingChartUpdates[playerKey] = false;
            } else {
                pendingChartUpdates[playerKey] = true;
            }

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

    const payoffSummary = payload.payoffs ?? DEFAULT_PAYOFFS;

    summaryContent.innerHTML = [
        renderSummaryMetric("Rounds", payload.rounds),
        renderSummaryMetric(
            "Player 1 Coins",
            payload.total_payoff.player1.toFixed(2)
        ),
        renderSummaryMetric(
            "Player 2 Coins",
            payload.total_payoff.player2.toFixed(2)
        ),
        renderSummaryMetric(
            "Player 1 Avg / Round",
            payload.average_payoff_per_round.player1.toFixed(3)
        ),
        renderSummaryMetric(
            "Player 2 Avg / Round",
            payload.average_payoff_per_round.player2.toFixed(3)
        ),
        renderSummaryMetric(
            "Player 1 Coop %",
            `${(payload.cooperation_rate.player1 * 100).toFixed(1)}%`
        ),
        renderSummaryMetric(
            "Player 2 Coop %",
            `${(payload.cooperation_rate.player2 * 100).toFixed(1)}%`
        ),
        renderSummaryMetric(
            "Reward (R)",
            payoffSummary.reward.toFixed(2)
        ),
        renderSummaryMetric(
            "Temptation (T)",
            payoffSummary.temptation.toFixed(2)
        ),
        renderSummaryMetric(
            "Sucker (S)",
            payoffSummary.sucker.toFixed(2)
        ),
        renderSummaryMetric(
            "Punishment (P)",
            payoffSummary.punishment.toFixed(2)
        ),
    ].join("");
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

    coinsAxisMax = 0;
    synchroniseCoinsAxes();
    resetPendingChartUpdates();
    resetPlayerStats();
    setSummaryPlaceholder();
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
                        precision: 0,
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
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Cooperation %",
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
                    grid: { color: "rgba(148, 163, 184, 0.08)" },
                },
                y: {
                    beginAtZero: true,
                    suggestedMax: 100,
                    max: 100,
                    ticks: {
                        color: "#cbd5f5",
                        precision: 0,
                        stepSize: 10,
                        callback: (value) => `${value}%`,
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

function appendChartPoint(chart, label, value) {
    chart.data.labels.push(label);
    chart.data.datasets[0].data.push(value);
    if (chart.data.labels.length > MAX_CHART_POINTS) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
    }
}

function shouldUpdateChart(roundNumber) {
    return roundNumber === 1 || roundNumber % CHART_UPDATE_INTERVAL === 0;
}

function updatePlayerCharts(playerCharts) {
    playerCharts.coins.update("none");
    playerCharts.cooperation.update("none");
}

function flushPendingChartUpdates(force = false) {
    if (!charts) {
        return;
    }
    PLAYER_KEYS.forEach((playerKey) => {
        if (!force && !pendingChartUpdates[playerKey]) {
            return;
        }
        const playerCharts = charts[playerKey];
        if (!playerCharts) {
            return;
        }
        updatePlayerCharts(playerCharts);
        pendingChartUpdates[playerKey] = false;
    });
}

function resetPendingChartUpdates() {
    PLAYER_KEYS.forEach((playerKey) => {
        pendingChartUpdates[playerKey] = false;
    });
}

function synchroniseCoinsAxes() {
    if (!charts) {
        return;
    }
    const baseMax = coinsAxisMax > 0 ? coinsAxisMax * 1.05 : 10;
    const paddedMax = Math.max(10, Math.ceil(baseMax));
    PLAYER_KEYS.forEach((playerKey) => {
        const playerCharts = charts[playerKey];
        if (!playerCharts) {
            return;
        }
        playerCharts.coins.options.scales.y.max = paddedMax;
        playerCharts.coins.options.scales.y.beginAtZero = true;
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

function setSummaryPlaceholder(message = "Run a simulation to see results.") {
    summaryContent.innerHTML = renderSummaryMetric("Status", message);
}

function setStatus(message, level = "info") {
    statusText.textContent = message;
    statusText.dataset.level = level;
}

function renderSummaryMetric(label, value) {
    return `<div class="metric"><span class="metric-label">${label}</span><span class="metric-value">${value}</span></div>`;
}

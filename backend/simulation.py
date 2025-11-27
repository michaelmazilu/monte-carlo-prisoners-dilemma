"""
Core simulation logic for the Monte Carlo Prisoner's Dilemma MVP.

The simulation is implemented with PyTorch tensors to enable efficient
vectorised tracking of payoffs and cooperation statistics. Each round
emits structured dictionaries that can be streamed to the frontend
through Server-Sent Events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Generator, List, Tuple

import torch
from torch import rand, randint


class SimulationValidationError(ValueError):
    """Raised when the simulation configuration is invalid."""


class StrategyLookupError(ValueError):
    """Raised when an unknown strategy key is provided."""


class StrategyType(str, Enum):
    """Supported strategy identifiers."""

    ALWAYS_COOPERATE = "always_cooperate"
    ALWAYS_DEFECT = "always_defect"
    PROBABILISTIC = "probabilistic"
    TIT_FOR_TAT = "tit_for_tat"
    RANDOM = "random"


ALLOWED_STRATEGY_KEYS = {
    StrategyType.ALWAYS_COOPERATE.value: StrategyType.ALWAYS_COOPERATE,
    StrategyType.ALWAYS_DEFECT.value: StrategyType.ALWAYS_DEFECT,
    StrategyType.PROBABILISTIC.value: StrategyType.PROBABILISTIC,
    StrategyType.TIT_FOR_TAT.value: StrategyType.TIT_FOR_TAT,
    StrategyType.RANDOM.value: StrategyType.RANDOM,
    "tit-for-tat": StrategyType.TIT_FOR_TAT,
}


DEFAULT_ROUND_EVENT_CHUNK_SIZE = 64


def resolve_strategy_type(key: str) -> StrategyType:
    try:
        return ALLOWED_STRATEGY_KEYS[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise StrategyLookupError(key) from exc


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for a single player's strategy."""

    strategy_type: StrategyType
    cooperate_probability: float = 1.0

    def __post_init__(self) -> None:
        if self.strategy_type is StrategyType.PROBABILISTIC:
            if not 0.0 <= self.cooperate_probability <= 1.0:
                raise SimulationValidationError(
                    "Probabilistic strategies require cooperate_probability in [0, 1]."
                )

    def sample_action(self, *, round_index: int, opponent_previous_action: int) -> int:
        """
        Draw an action for the current round.

        Returns 0 for cooperate and 1 for defect.
        """
        if self.strategy_type is StrategyType.ALWAYS_COOPERATE:
            return 0
        if self.strategy_type is StrategyType.ALWAYS_DEFECT:
            return 1
        if self.strategy_type is StrategyType.TIT_FOR_TAT:
            if round_index == 1:
                return 0
            return int(bool(opponent_previous_action))
        if self.strategy_type is StrategyType.RANDOM:
            return int(randint(0, 2, (1,), dtype=torch.int64).item())
        # Probabilistic strategy
        cooperate = rand(()) < self.cooperate_probability
        return 0 if bool(cooperate.item()) else 1  # 0 => cooperate, 1 => defect


@dataclass(frozen=True)
class PayoffConfig:
    """Numerical values that define the payoff matrix."""

    reward: float = 3.0
    temptation: float = 5.0
    sucker: float = 0.0
    punishment: float = 1.0

    def to_tensor(self) -> torch.Tensor:
        """Return the 2x2 payoff matrix for the configured values."""
        return torch.tensor(
            [
                [[self.reward, self.reward], [self.sucker, self.temptation]],
                [[self.temptation, self.sucker], [self.punishment, self.punishment]],
            ],
            dtype=torch.float32,
        )


@dataclass(frozen=True)
class SimulationConfig:
    """Complete configuration for a simulation run."""

    rounds: int
    monte_carlo_runs: int
    player_strategies: Tuple[StrategyConfig, StrategyConfig]
    payoffs: PayoffConfig = field(default_factory=PayoffConfig)
    round_event_chunk_size: int = DEFAULT_ROUND_EVENT_CHUNK_SIZE

    def __post_init__(self) -> None:
        if self.rounds <= 0:
            raise SimulationValidationError("Number of rounds must be positive.")
        if self.monte_carlo_runs <= 0:
            raise SimulationValidationError("Monte Carlo runs must be positive.")
        if self.round_event_chunk_size <= 0:
            raise SimulationValidationError("Round event chunk size must be positive.")


OUTCOME_KEYS = ("CC", "CD", "DC", "DD")
OUTCOME_INDEX = {
    (0, 0): 0,  # CC
    (0, 1): 1,  # CD
    (1, 0): 2,  # DC
    (1, 1): 3,  # DD
}

def _format_tensor(values: torch.Tensor) -> Tuple[float, ...]:
    """Convert a 1D tensor into a tuple of floats."""
    return tuple(float(x) for x in values.tolist())


def _format_counts(counts: torch.Tensor) -> Dict[str, int]:
    """Convert outcome counts into a named dictionary."""
    return {key: int(counts[idx].item()) for idx, key in enumerate(OUTCOME_KEYS)}


def run_simulation(
    config: SimulationConfig,
) -> Generator[Tuple[str, Dict[str, object]], None, None]:
    """
    Execute the simulation and yield structured events suitable for SSE.

    Yields:
        Tuples of (event_name, payload_dict)
    """

    total_rounds = config.rounds
    total_runs = config.monte_carlo_runs
    payoff_matrix = config.payoffs.to_tensor()
    chunk_size = config.round_event_chunk_size

    overall_payoff = torch.zeros(2, dtype=torch.float32)
    overall_cooperation_counts = torch.zeros(2, dtype=torch.float32)
    overall_outcome_counts = torch.zeros(len(OUTCOME_KEYS), dtype=torch.float32)

    for run_index in range(1, total_runs + 1):
        run_payoff = torch.zeros(2, dtype=torch.float32)
        run_cooperation_counts = torch.zeros(2, dtype=torch.float32)
        run_outcome_counts = torch.zeros(len(OUTCOME_KEYS), dtype=torch.float32)
        previous_actions = (0, 0)
        round_buffer: List[Dict[str, object]] = []

        for round_index in range(1, total_rounds + 1):
            action_player1 = config.player_strategies[0].sample_action(
                round_index=round_index,
                opponent_previous_action=previous_actions[1],
            )
            action_player2 = config.player_strategies[1].sample_action(
                round_index=round_index,
                opponent_previous_action=previous_actions[0],
            )
            payoff = payoff_matrix[action_player1, action_player2]
            run_payoff[0] += payoff[0]
            run_payoff[1] += payoff[1]
            if action_player1 == 0:
                run_cooperation_counts[0] += 1.0
            if action_player2 == 0:
                run_cooperation_counts[1] += 1.0
            outcome_idx = OUTCOME_INDEX[(action_player1, action_player2)]
            run_outcome_counts[outcome_idx] += 1.0

            cumulative_round = (run_index - 1) * total_rounds + round_index
            cooperated_flags = (action_player1 == 0, action_player2 == 0)
            round_payload = {
                "run": run_index,
                "round": round_index,
                "cumulative_round": cumulative_round,
                "actions": {
                    "player1": "C" if action_player1 == 0 else "D",
                    "player2": "C" if action_player2 == 0 else "D",
                },
                "cooperated": {
                    "player1": bool(cooperated_flags[0]),
                    "player2": bool(cooperated_flags[1]),
                },
                "cumulative_cooperation": {
                    "player1": int(run_cooperation_counts[0].item()),
                    "player2": int(run_cooperation_counts[1].item()),
                },
                "round_payoff": {
                    "player1": float(payoff[0].item()),
                    "player2": float(payoff[1].item()),
                },
                "total_payoff": {
                    "player1": float(run_payoff[0].item()),
                    "player2": float(run_payoff[1].item()),
                },
                "cooperation_rate": {
                    "player1": float(run_cooperation_counts[0].item() / round_index),
                    "player2": float(run_cooperation_counts[1].item() / round_index),
                },
                "outcome_counts": _format_counts(run_outcome_counts),
            }

            round_buffer.append(round_payload)
            if len(round_buffer) >= chunk_size:
                yield ("round_batch", {"rounds": round_buffer})
                round_buffer = []

            previous_actions = (action_player1, action_player2)

        overall_payoff += run_payoff
        overall_cooperation_counts += run_cooperation_counts
        overall_outcome_counts += run_outcome_counts

        if round_buffer:
            yield ("round_batch", {"rounds": round_buffer})

        yield (
            "run_complete",
            {
                "run": run_index,
                "total_payoff": {
                    "player1": float(run_payoff[0].item()),
                    "player2": float(run_payoff[1].item()),
                },
                "total_cooperation": {
                    "player1": int(run_cooperation_counts[0].item()),
                    "player2": int(run_cooperation_counts[1].item()),
                },
                "average_payoff_per_round": {
                    "player1": float(run_payoff[0].item() / total_rounds),
                    "player2": float(run_payoff[1].item() / total_rounds),
                },
                "cooperation_rate": {
                    "player1": float(run_cooperation_counts[0].item() / total_rounds),
                    "player2": float(run_cooperation_counts[1].item() / total_rounds),
                },
                "outcome_counts": _format_counts(run_outcome_counts),
            },
        )

    total_rounds_played = float(total_rounds * total_runs)
    final_summary = {
        "runs": total_runs,
        "rounds": total_rounds,
        "total_payoff": {
            "player1": float(overall_payoff[0].item()),
            "player2": float(overall_payoff[1].item()),
        },
        "average_payoff_per_round": {
            "player1": float(overall_payoff[0].item() / total_rounds_played),
            "player2": float(overall_payoff[1].item() / total_rounds_played),
        },
        "cooperation_rate": {
            "player1": float(overall_cooperation_counts[0].item() / total_rounds_played),
            "player2": float(overall_cooperation_counts[1].item() / total_rounds_played),
        },
        "total_cooperation": {
            "player1": int(overall_cooperation_counts[0].item()),
            "player2": int(overall_cooperation_counts[1].item()),
        },
        "outcome_counts": _format_counts(overall_outcome_counts),
        "outcome_distribution": {
            key: float(overall_outcome_counts[idx].item() / total_rounds_played)
            for idx, key in enumerate(OUTCOME_KEYS)
        },
        "payoffs": {
            "reward": float(config.payoffs.reward),
            "temptation": float(config.payoffs.temptation),
            "sucker": float(config.payoffs.sucker),
            "punishment": float(config.payoffs.punishment),
        },
        "round_event_chunk_size": chunk_size,
    }

    yield ("summary", final_summary)


def serialize_config(config: SimulationConfig) -> str:
    """Serialize the configuration for debugging or logging."""
    data = {
        "rounds": config.rounds,
        "monte_carlo_runs": config.monte_carlo_runs,
        "strategies": [
            {
                "strategy_type": config.player_strategies[0].strategy_type.value,
                "cooperate_probability": config.player_strategies[0].cooperate_probability,
            },
            {
                "strategy_type": config.player_strategies[1].strategy_type.value,
                "cooperate_probability": config.player_strategies[1].cooperate_probability,
            },
        ],
        "payoffs": {
            "reward": config.payoffs.reward,
            "temptation": config.payoffs.temptation,
            "sucker": config.payoffs.sucker,
            "punishment": config.payoffs.punishment,
        },
        "round_event_chunk_size": config.round_event_chunk_size,
    }
    return json.dumps(data, sort_keys=True)

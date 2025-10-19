import unittest

import torch

from backend.simulation import (
    SimulationConfig,
    SimulationValidationError,
    StrategyConfig,
    StrategyType,
    PayoffConfig,
    run_simulation,
)


class SimulationConfigTests(unittest.TestCase):
    def test_invalid_rounds_raise(self):
        with self.assertRaises(SimulationValidationError):
            SimulationConfig(
                rounds=0,
                monte_carlo_runs=1,
                player_strategies=(
                    StrategyConfig(StrategyType.ALWAYS_COOPERATE),
                    StrategyConfig(StrategyType.ALWAYS_COOPERATE),
                ),
            )

    def test_invalid_probability_raises(self):
        with self.assertRaises(SimulationValidationError):
            StrategyConfig(StrategyType.PROBABILISTIC, cooperate_probability=1.5)

    def test_cooperate_strategy_produces_expected_totals(self):
        config = SimulationConfig(
            rounds=3,
            monte_carlo_runs=1,
            player_strategies=(
                StrategyConfig(StrategyType.ALWAYS_COOPERATE),
                StrategyConfig(StrategyType.ALWAYS_COOPERATE),
            ),
        )

        events = list(run_simulation(config))
        round_events = [payload for event, payload in events if event == "round"]
        self.assertEqual(len(round_events), 3)
        first_round = round_events[0]
        self.assertIn("cooperated", first_round)
        self.assertTrue(first_round["cooperated"]["player1"])
        self.assertTrue(first_round["cooperated"]["player2"])
        self.assertIn("cumulative_cooperation", first_round)
        self.assertEqual(first_round["cumulative_cooperation"]["player1"], 1)
        self.assertEqual(first_round["cumulative_cooperation"]["player2"], 1)

        summary = next(payload for event, payload in events if event == "summary")
        self.assertAlmostEqual(summary["total_payoff"]["player1"], 9.0)
        self.assertAlmostEqual(summary["total_payoff"]["player2"], 9.0)
        self.assertAlmostEqual(summary["average_payoff_per_round"]["player1"], 3.0)
        self.assertAlmostEqual(summary["cooperation_rate"]["player1"], 1.0)
        self.assertEqual(summary["total_cooperation"]["player1"], 3)
        self.assertEqual(summary["total_cooperation"]["player2"], 3)

    def test_probabilistic_strategy_repeatable_with_seed(self):
        torch.manual_seed(42)
        config = SimulationConfig(
            rounds=5,
            monte_carlo_runs=1,
            player_strategies=(
                StrategyConfig(StrategyType.PROBABILISTIC, cooperate_probability=0.75),
                StrategyConfig(StrategyType.ALWAYS_DEFECT),
            ),
        )

        events = [payload for event, payload in run_simulation(config) if event == "round"]
        actions = [event_payload["actions"]["player1"] for event_payload in events]
        self.assertEqual(actions.count("C"), 4)
        self.assertEqual(actions.count("D"), 1)

    def test_tit_for_tat_vs_defect_behaviour(self):
        config = SimulationConfig(
            rounds=4,
            monte_carlo_runs=1,
            player_strategies=(
                StrategyConfig(StrategyType.TIT_FOR_TAT),
                StrategyConfig(StrategyType.ALWAYS_DEFECT),
            ),
        )

        events = list(run_simulation(config))
        summary = next(payload for event, payload in events if event == "summary")
        self.assertAlmostEqual(summary["total_payoff"]["player1"], 3.0)
        self.assertAlmostEqual(summary["total_payoff"]["player2"], 8.0)
        rounds = [payload for event, payload in events if event == "round"]
        self.assertEqual(rounds[0]["actions"]["player1"], "C")
        for round_payload in rounds[1:]:
            self.assertEqual(round_payload["actions"]["player1"], "D")

    def test_custom_payoff_values_are_respected(self):
        payoffs = PayoffConfig(reward=4.0, temptation=9.0, sucker=-2.0, punishment=0.5)
        config = SimulationConfig(
            rounds=2,
            monte_carlo_runs=1,
            player_strategies=(
                StrategyConfig(StrategyType.ALWAYS_COOPERATE),
                StrategyConfig(StrategyType.ALWAYS_COOPERATE),
            ),
            payoffs=payoffs,
        )

        events = list(run_simulation(config))
        summary = next(payload for event, payload in events if event == "summary")
        self.assertAlmostEqual(summary["total_payoff"]["player1"], 8.0)
        self.assertAlmostEqual(summary["total_payoff"]["player2"], 8.0)
        self.assertIn("payoffs", summary)
        self.assertAlmostEqual(summary["payoffs"]["reward"], payoffs.reward)
        self.assertAlmostEqual(summary["payoffs"]["temptation"], payoffs.temptation)
        self.assertAlmostEqual(summary["payoffs"]["sucker"], payoffs.sucker)
        self.assertAlmostEqual(summary["payoffs"]["punishment"], payoffs.punishment)


if __name__ == "__main__":
    unittest.main()

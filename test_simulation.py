import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from app import run_monte_carlo_simulation

class TestMonteCarloSimulation(unittest.TestCase):
    
    def test_probabilistic_strategy(self):
        """Test probabilistic strategy simulation"""
        result = run_monte_carlo_simulation(0.5, 0.5, 1000, "probabilistic", "probabilistic")
        
        # Check that all required fields are present
        self.assertIn('player1_avg_payoff', result)
        self.assertIn('player2_avg_payoff', result)
        self.assertIn('player1_coop_rate', result)
        self.assertIn('player2_coop_rate', result)
        self.assertIn('both_cooperate_rate', result)
        self.assertIn('both_defect_rate', result)
        
        # Check that cooperation rates are reasonable (should be close to input probabilities)
        self.assertAlmostEqual(result['player1_coop_rate'], 0.5, delta=0.1)
        self.assertAlmostEqual(result['player2_coop_rate'], 0.5, delta=0.1)
        
        # Check that payoffs are within expected range (0-5)
        self.assertGreaterEqual(result['player1_avg_payoff'], 0)
        self.assertLessEqual(result['player1_avg_payoff'], 5)
        self.assertGreaterEqual(result['player2_avg_payoff'], 0)
        self.assertLessEqual(result['player2_avg_payoff'], 5)
    
    def test_always_cooperate_strategy(self):
        """Test always cooperate strategy"""
        result = run_monte_carlo_simulation(0.5, 0.5, 1000, "always_cooperate", "always_cooperate")
        
        # Both players should always cooperate
        self.assertEqual(result['player1_coop_rate'], 1.0)
        self.assertEqual(result['player2_coop_rate'], 1.0)
        self.assertEqual(result['both_cooperate_rate'], 1.0)
        self.assertEqual(result['both_defect_rate'], 0.0)
        
        # Both should get payoff of 3 (mutual cooperation)
        self.assertEqual(result['player1_avg_payoff'], 3.0)
        self.assertEqual(result['player2_avg_payoff'], 3.0)
    
    def test_always_defect_strategy(self):
        """Test always defect strategy"""
        result = run_monte_carlo_simulation(0.5, 0.5, 1000, "always_defect", "always_defect")
        
        # Both players should never cooperate
        self.assertEqual(result['player1_coop_rate'], 0.0)
        self.assertEqual(result['player2_coop_rate'], 0.0)
        self.assertEqual(result['both_cooperate_rate'], 0.0)
        self.assertEqual(result['both_defect_rate'], 1.0)
        
        # Both should get payoff of 1 (mutual defection)
        self.assertEqual(result['player1_avg_payoff'], 1.0)
        self.assertEqual(result['player2_avg_payoff'], 1.0)
    
    def test_mixed_strategies(self):
        """Test mixed strategies"""
        result = run_monte_carlo_simulation(0.5, 0.5, 1000, "always_cooperate", "always_defect")
        
        # Player 1 always cooperates, Player 2 always defects
        self.assertEqual(result['player1_coop_rate'], 1.0)
        self.assertEqual(result['player2_coop_rate'], 0.0)
        self.assertEqual(result['both_cooperate_rate'], 0.0)
        self.assertEqual(result['both_defect_rate'], 0.0)
        self.assertEqual(result['p1_coop_p2_defect_rate'], 1.0)
        self.assertEqual(result['p1_defect_p2_coop_rate'], 0.0)
        
        # Player 1 gets 0 (sucker's payoff), Player 2 gets 5 (temptation payoff)
        self.assertEqual(result['player1_avg_payoff'], 0.0)
        self.assertEqual(result['player2_avg_payoff'], 5.0)
    
    def test_outcome_rates_sum_to_one(self):
        """Test that all outcome rates sum to 1"""
        result = run_monte_carlo_simulation(0.3, 0.7, 1000, "probabilistic", "probabilistic")
        
        total_rate = (result['both_cooperate_rate'] + 
                     result['both_defect_rate'] + 
                     result['p1_coop_p2_defect_rate'] + 
                     result['p1_defect_p2_coop_rate'])
        
        self.assertAlmostEqual(total_rate, 1.0, places=5)
    
    def test_large_simulation(self):
        """Test simulation with large number of rounds"""
        result = run_monte_carlo_simulation(0.5, 0.5, 10000, "probabilistic", "probabilistic")
        
        # With more rounds, cooperation rates should be closer to input probabilities
        self.assertAlmostEqual(result['player1_coop_rate'], 0.5, delta=0.05)
        self.assertAlmostEqual(result['player2_coop_rate'], 0.5, delta=0.05)
        
        # Check that total rounds is correct
        self.assertEqual(result['total_rounds'], 10000)

if __name__ == '__main__':
    unittest.main()

import sys
import os
import unittest
import pandas as pd

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from market_scanner.features import FeatureEngineer

class TestMomentumFeatures(unittest.TestCase):
    def setUp(self):
        self.fe = FeatureEngineer()

    def test_calculate_momentum_burst_ignition(self):
        quote_data = {'volume': 1000, 'oi_change': 100} # RVol = 10 (Ignition)
        res = self.fe.calculate_momentum_burst(quote_data)
        self.assertEqual(res['status'], "IGNITION")
        self.assertEqual(res['rvol_score'], 10.0)

    def test_calculate_momentum_burst_active(self):
        quote_data = {'volume': 200, 'oi_change': 100} # RVol = 2 (Active)
        res = self.fe.calculate_momentum_burst(quote_data)
        self.assertEqual(res['status'], "ACTIVE")
        self.assertEqual(res['rvol_score'], 2.0)

    def test_calculate_gamma_decoupling_undervalued(self):
        # Spot up 1%, Delta 0.5 -> Expected option move = 0.5% * price
        # If option price stays same, it's undervalued.
        spot_price = 10100
        prev_spot = 10000
        option_price = 100
        prev_option_price = 100
        delta = 0.5
        
        # Spot_Change_Pct = 0.01
        # Expected_Move = (0.01 * 0.5) * 100 = 0.5
        # Actual_Move = 0
        # Dislocation = -0.5
        # Dislocation_Pct = -0.5% (Hmm, needs to be > -5% for signal in my current code)
        
        # Let's make it a bigger move
        spot_price = 12000
        prev_spot = 10000 # 20% move
        # Expected move = 20% * 0.5 = 10% move on option
        # Expected price = 110
        option_price = 100 # Actual price stayed at 100
        
        res = self.fe.calculate_gamma_decoupling(spot_price, prev_spot, option_price, prev_option_price, delta)
        self.assertEqual(res['signal'], "Undervalued (Buy Signal)")
        self.assertLess(res['dislocation_pct'], -5.0)

if __name__ == '__main__':
    unittest.main()

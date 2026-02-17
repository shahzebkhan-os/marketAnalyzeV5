import pandas as pd
import numpy as np
from features import FeatureEngineer

def test_antigravity_logic():
    fe = FeatureEngineer()
    
    # EXTREME Short Squeeze Scenario
    # Spot is 25050
    # Wall is at 25100
    # Wall breach: 25050 >= 25100 * 0.995 (24974.5) -> True (40 pts)
    # Panic: Massive OI shed near spot
    
    data = [
        {"strike": 25100, "type": "CE", "oi": 100000, "oi_change": -15000, "iv": 22.0, "ltp_change": 10.0},
        {"strike": 25000, "type": "CE", "oi": 50000, "oi_change": -10000, "iv": 23.0, "ltp_change": 8.0},
        {"strike": 25050, "type": "PE", "oi": 30000, "oi_change": 5000, "iv": 20.0, "ltp_change": -5.0} # Put writing (bullish)
    ]
    df = pd.DataFrame(data)
    spot = 25050
    
    result = fe.calculate_antigravity(df, spot)
    print(f"--- Extreme Short Squeeze Scenario ---")
    print(f"Result: {result}")
    
    # Trap: 40 pts
    # Panic: oi_change_sum = -25000. total_near_oi = 150000. ratio = 0.166. 
    # panic_score = min(40, (0.166/0.1)*40) = 40 pts.
    # Velocity: 20 pts (ltp_change > 0)
    # Total: 100 pts
    
    assert result['score'] >= 80
    assert result['status'] == True
    assert result['wall_strike'] == 25100
    assert result['oi_shed'] == 25000
    
    print("Test Passed!")

if __name__ == "__main__":
    test_antigravity_logic()

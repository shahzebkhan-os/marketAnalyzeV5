import pytest
from datetime import datetime, timedelta
from shock_detector import ShockDetector

def test_price_velocity_shock():
    sd = ShockDetector()
    base_time = datetime.now()
    
    # 1. Normal data
    sd.update("NIFTY", 100.0, 15.0, 1.0, base_time)
    res = sd.update("NIFTY", 100.1, 15.0, 1.0, base_time + timedelta(seconds=10))
    assert not res['triggered']
    
    # 2. Add enough history
    for i in range(15):
        sd.update("NIFTY", 100.0, 15.0, 1.0, base_time + timedelta(seconds=10*i))
        
    # 3. Sudden Jump (> 0.7%)
    # 100 * 1.008 = 100.8
    res = sd.update("NIFTY", 100.8, 15.0, 1.0, base_time + timedelta(seconds=70))
    
    assert res['triggered']
    assert res['type'] == 'PRICE_MOVE'

def test_iv_spike_shock():
    sd = ShockDetector()
    base_time = datetime.now()
    
    # Fill history with stable IV
    for i in range(20):
        sd.update("NIFTY", 100.0, 15.0, 1.0, base_time + timedelta(seconds=10*i))
        
    # Spike IV (Mean ~15, Std ~0 -> Spike to 20 should trigger if std is small?)
    # If std is 0, z-score might be div by zero?
    # Actually code checks `if iv_std > 0`.
    # Let's add some noise to make std > 0
    sd.update("NIFTY", 100.0, 15.1, 1.0, base_time + timedelta(seconds=200))
    
    # Now huge spike
    res = sd.update("NIFTY", 100.0, 25.0, 1.0, base_time + timedelta(seconds=210))
    
    # With mean ~15 and small std, 25 is huge z-score
    assert res['triggered']
    assert res['type'] == 'IV_SPIKE'

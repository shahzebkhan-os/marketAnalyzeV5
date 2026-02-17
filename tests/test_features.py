import pytest
import pandas as pd
import numpy as np
from features import FeatureEngineer

@pytest.fixture
def engineer():
    return FeatureEngineer()

def test_compute_oi_stats(engineer):
    chain_data = [
        {"strike": 10000, "call_oi": 100, "put_oi": 50},
        {"strike": 10100, "call_oi": 50, "put_oi": 100},
        {"strike": 10200, "call_oi": 10, "put_oi": 10},
    ]
    stats = engineer.compute_oi_stats(chain_data)
    
    assert "pcr" in stats
    assert "oi_imbalance" in stats
    assert "max_pain" in stats
    
    total_call = 160
    total_put = 160
    assert stats["pcr"] == 1.0
    assert stats["oi_imbalance"] == 0

def test_compute_greeks_black_scholes(engineer):
    # Test BS calculator
    # S=100, K=100, T=1, r=0.05, sigma=0.2
    # Call Delta should be approx 0.6368
    delta, gamma = engineer.black_scholes_greeks(100, 100, 1, 0.05, 0.2, 'call')
    assert 0.6 < delta < 0.7
    assert gamma > 0

def test_compute_technicals_empty(engineer):
    stats = engineer.compute_technicals([])
    # Should probably return valid dict with 0s or empty
    assert isinstance(stats, dict)
    # Based on implementation, if empty it returns {} or logs error and returns {}
    # Let's check implementation behavior
    # It returns {} on exception/empty
    assert stats == {}

def test_compute_technicals_valid(engineer):
    prices = [100 + i for i in range(60)] # 60 data points
    stats = engineer.compute_technicals(prices)
    
    assert "ema20" in stats
    assert "rsi" in stats
    assert "macd" in stats
    assert stats["ema20"] > 0
    assert stats["rsi"] > 0 # Simple uptrend, RSI should be high

def test_deterministic_seed(engineer):
    from features import set_seed
    # Verify numpy random numbers are same
    np.random.seed(42)
    val1 = np.random.rand()
    
    set_seed(42) # should reset
    val2 = np.random.rand()
    
    assert val1 == val2
    
    # Verify FeatureEngineer init resets it
    FeatureEngineer()
    val3 = np.random.rand()
    # Since FeatureEngineer calls set_seed(42) in init, val3 should match the first value 
    # generated after seed(42). 
    # Wait, val1 was the first value. val2 was first value. val3 should be first value.
    assert val1 == val3

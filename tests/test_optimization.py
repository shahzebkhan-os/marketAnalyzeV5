import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import MagicMock
from optimizer import HyperparameterOptimizer

class MockBacktester:
    def __init__(self):
        self.stop_loss_pct = 0.01

    def run(self, data, signals):
        # Do nothing, just mock
        pass
    
    def calculate_metrics(self):
        # Return synthetic metrics
        # We can add some randomness or dependence on instance vars if we want robust checks
        # For now just return a valid shape
        return {'Sharpe Ratio': 1.5, 'Max Drawdown': -0.1}

def test_optimization_flow():
    # Mock Data
    dates = pd.date_range("2024-01-01", periods=100, freq="H")
    data = pd.DataFrame({
        "close": np.random.rand(100) * 100
    }, index=dates)
    
    base_signals = pd.DataFrame({
        "lgbm": np.random.rand(100),
        "xgb": np.random.rand(100),
        "lstm": np.random.rand(100)
    })
    
    # Use SQLite in memory or temp file
    db_url = "sqlite:///test_optuna.db"
    if os.path.exists("test_optuna.db"):
        os.remove("test_optuna.db")

    optimizer = HyperparameterOptimizer(MockBacktester, data, base_signals, db_url=db_url)
    
    # Run 1 trial to be fast
    study = optimizer.run_optimization(n_trials=2)
    
    assert "weight_lgbm" in study
    assert "confidence_threshold" in study
    
    # Check history
    df = optimizer.get_optimization_history()
    assert not df.empty
    assert len(df) >= 2
    
    # Cleanup
    if os.path.exists("test_optuna.db"):
        os.remove("test_optuna.db")

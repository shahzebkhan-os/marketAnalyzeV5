import pytest
import numpy as np
import pandas as pd
from drift import DriftDetector
from alerts import AlertManager

class MockAlertManager(AlertManager):
    def __init__(self):
        super().__init__()
        self.sent_messages = []
        
    def send_message(self, text: str, priority: str = "INFO"):
        self.sent_messages.append((priority, text))

def test_psi_computation():
    dd = DriftDetector()
    
    # Identical distributions -> Low PSI
    d1 = np.random.normal(0, 1, 1000)
    d2 = d1.copy()
    psi = dd.compute_psi(d1, d2)
    assert psi < 0.1
    
    # Shifted distributions -> High PSI
    d3 = np.random.normal(2, 1, 1000)
    psi_shifted = dd.compute_psi(d1, d3)
    assert psi_shifted > 0.25

def test_check_drift():
    am = MockAlertManager()
    dd = DriftDetector(alert_manager=am)
    
    # Baseline
    df_base = pd.DataFrame({
        "feat1": np.random.normal(0, 1, 100)
    })
    dd.set_baseline(df_base)
    
    # No Drift
    df_curr = pd.DataFrame({
        "feat1": np.random.normal(0, 1, 100)
    })
    logs = dd.check_drift(df_curr)
    assert len(logs) == 0 # Or logs with OK status depending on impl details, but impl only returns logs on alert
    
    # Drift
    df_drift = pd.DataFrame({
        "feat1": np.random.normal(5, 1, 100)
    })
    logs = dd.check_drift(df_drift)
    assert len(logs) > 0
    assert logs[0]['status'] in ["CRITICAL", "WARNING"]
    assert len(am.sent_messages) > 0

def test_safety_mechanism():
    am = MockAlertManager()
    dd = DriftDetector(alert_manager=am)
    
    assert dd.is_model_enabled is True
    
    # Simulate Good Performance
    for _ in range(30):
        dd.update_rolling_auc(1, 0.9) # Perfect pres
        dd.update_rolling_auc(0, 0.1) # Perfect preds
    
    assert dd.is_model_enabled is True
    
    # Simulate Bad Performance (AUC < 0.55)
    # We need to feed data that results in low AUC.
    # Inverse labels: Say 1, Prob 0.1. Say 0, Prob 0.9.
    for _ in range(50):
        dd.update_rolling_auc(1, 0.1) 
        dd.update_rolling_auc(0, 0.9)
    
    # Should trigger safety stop
    assert dd.is_model_enabled is False
    assert any("SAFETY STOP" in msg[1] for msg in am.sent_messages)

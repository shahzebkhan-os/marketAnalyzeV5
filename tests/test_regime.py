import pytest
import pandas as pd
import numpy as np
import shutil
import os
from regime import RegimeDetector

@pytest.fixture
def detector():
    test_dir = "test_regime_models"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    det = RegimeDetector(models_dir=test_dir, history_window=50)
    yield det
    
    # Cleanup
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

def test_fit_and_detect(detector):
    # Create dummy data
    # 100 points
    np.random.seed(42)
    closes = np.cumprod(1 + np.random.normal(0, 0.01, 100)) * 100
    atr = np.random.uniform(0.5, 2.0, 100)
    iv = np.random.uniform(10, 30, 100)
    
    df = pd.DataFrame({
        "close": closes,
        "atr": atr,
        "iv": iv
    })
    
    # Fit
    detector.fit_models(df)
    
    assert detector.hmm_model is not None
    assert detector.kmeans_model is not None
    
    # Detect
    # Create a scenario for "High ATR" -> Explosive
    df_explosive = df.copy()
    feature_expl = pd.DataFrame({
        "close": [110],
        "atr": [5.0], # Very high compared to history
        "iv": [40]
    })
    # We need to append to history to calculate percentile
    df_combined = pd.concat([df_explosive, feature_expl], ignore_index=True)
    
    # Recalculate returns/vol for the new row implicitly in detect_regime?
    # detect_regime calculates returns/vol if missing.
    
    regime = detector.detect_regime(df_combined)
    assert regime['label'] == "Explosive"
    assert "hmm_state" in regime['details']
    
    # Create scenario for Low ATR -> Ranging
    feature_range = pd.DataFrame({
        "close": [110],
        "atr": [0.1], # Very low
        "iv": [10]
    })
    df_range = pd.concat([df_explosive, feature_range], ignore_index=True)
    regime = detector.detect_regime(df_range)
    # 0.1 is likely < 0.3 percentile
    assert regime['label'] == "Ranging"

def test_save_load(detector):
    # Dummy fit
    df = pd.DataFrame({
        "close": np.random.rand(60),
        "atr": np.random.rand(60),
        "iv": np.random.rand(60)
    })
    detector.fit_models(df)
    
    # New detector instance
    det2 = RegimeDetector(models_dir=detector.models_dir)
    assert det2.hmm_model is not None

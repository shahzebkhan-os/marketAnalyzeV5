import pytest
import pandas as pd
import numpy as np
import os
import shutil
from ensemble import StackingEnsemble

@pytest.fixture
def ensemble():
    test_dir = "test_ensemble_models"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    ens = StackingEnsemble(models_dir=test_dir, penalty_factor=1.0)
    yield ens
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

def test_train_and_predict(ensemble):
    # Create dummy meta-features
    # Cols: lgbm, xgb, logreg, lstm, regime, variance
    n = 50
    X = pd.DataFrame({
        "lgbm": np.random.rand(n),
        "xgb": np.random.rand(n),
        "logreg": np.random.rand(n),
        "lstm": np.random.rand(n),
        "regime": np.random.randint(0, 3, n),
        "variance": np.random.uniform(0, 0.1, n)
    })
    y = pd.Series(np.random.rand(n)) # Target: actual probability/outcome
    
    ensemble.train_stacker(X, y)
    assert ensemble.stacker is not None
    assert os.path.exists(ensemble.model_path)

    # Predict
    base_preds = {"lgbm": 0.8, "xgb": 0.8, "logreg": 0.7, "lstm": 0.9}
    regime = {"label": "Trending"} # maps to 1
    var = 0.05
    
    result = ensemble.predict(base_preds, regime, var)
    assert 0 <= result['final_confidence'] <= 100
    assert result['variance'] == var
    
    # Test Penalization
    # High variance should lower confidence
    var_high = 0.5
    result_high_var = ensemble.predict(base_preds, regime, var_high)
    
    # Since inputs are same except variance, pre-penalty prob should be similar (actually variance is an input feature too, so stacker might react to it)
    # But the explicit penalty is subtracted at the end.
    # penalty = penalty_factor * variance. 
    # With factor=1.0, penalty is 0.5 vs 0.05.
    # So confident should drop significantly.
    
    assert result_high_var['final_confidence'] < result['final_confidence']

def test_predict_without_model(ensemble):
    base_preds = {"lgbm": 0.6, "xgb": 0.6}
    regime = {"label": "Ranging"}
    var = 0.1
    
    # No train called
    res = ensemble.predict(base_preds, regime, var)
    assert res['final_confidence'] == 60.0 # Average
    assert res['reason'] == "Average (No Stacker)"

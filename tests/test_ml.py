import pytest
import shutil
import os
import json
import pandas as pd
import numpy as np
from ml_engine import ModelTrainer
from sklearn.linear_model import LogisticRegression

@pytest.fixture
def trainer():
    test_dir = "test_models"
    registry = "test_models/registry.json"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    trainer = ModelTrainer(registry_path=registry, models_dir=test_dir)
    yield trainer
    
    # Cleanup
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

def test_train_models(trainer):
    # Create dummy data
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100))
    
    # Train
    results = trainer.train_models(X, y)
    
    assert "lgbm" in results
    assert "xgb" in results
    assert "logreg" in results
    
    assert results["lgbm"]["auc"] >= 0 # AUC can be 0 if terrible but usually > 0
    assert "importance" in results["lgbm"]

def test_save_model_registry(trainer):
    # Create dummy data
    X = pd.DataFrame(np.random.rand(20, 2), columns=["a", "b"])
    y = pd.Series(np.random.randint(0, 2, 20))
    
    # Create a dummy model (we can just use any object for pickle test, but let's use a real one)
    model = LogisticRegression()
    model.fit(X, y)
    
    model_data = {
        "model": model,
        "auc": 0.85,
        "feature_names": ["a", "b"],
        "importance": {"a": 0.1, "b": 0.9}
    }
    
    trainer.save_model("test_model", model_data)
    
    # Check file exists
    assert len(os.listdir(trainer.models_dir)) == 2 # registry.json + 1 pkl
    
    # Check registry
    with open(trainer.registry_path, 'r') as f:
        registry = json.load(f)
    
    assert "test_model" in registry
    entry = registry["test_model"][0]
    assert entry["auc"] == 0.85
    assert entry["importance"]["b"] == 0.9

def test_determinism(trainer):
    # Same data, same seed -> same result
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(50, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 50))
    
    trainer.set_seed(123)
    results1 = trainer.train_models(X, y)
    auc1 = results1["lgbm"]["auc"]
    
    trainer.set_seed(123)
    results2 = trainer.train_models(X, y)
    auc2 = results2["lgbm"]["auc"]
    
    assert auc1 == auc2

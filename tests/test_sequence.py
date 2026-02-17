import pytest
import pandas as pd
import numpy as np
import os
import shutil
import torch
from sequence_model import SequenceTrainer, OptionSequenceModel

@pytest.fixture
def trainer():
    test_dir = "test_seq_models"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    t = SequenceTrainer(models_dir=test_dir, seq_length=10)
    yield t
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

def test_data_prep_and_train(trainer):
    # Dummy data
    n_samples = 100
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="min"),
        "pcr": np.random.uniform(0.5, 1.5, n_samples),
        "gex": np.random.uniform(-100, 100, n_samples),
        "max_pain": np.random.uniform(10000, 20000, n_samples),
        "iv_skew": np.random.uniform(-5, 5, n_samples)
    })
    
    target = pd.Series(np.random.randint(0, 2, n_samples))
    
    # Check data prep
    X, y = trainer.prepare_data(df, target)
    assert X.shape[0] == n_samples - trainer.seq_length
    assert X.shape[1] == trainer.seq_length
    assert X.shape[2] == trainer.input_dim
    assert y.shape[0] == X.shape[0]
    
    # Train
    trainer.train_model(df, target, epochs=2, batch_size=8)
    assert os.path.exists(trainer.model_path)
    
    # Predict MC Dropout
    sample_input = X[0:1] # (1, seq_len, input_dim)
    mean, std = trainer.predict_mc_dropout(sample_input, n_samples=5)
    
    assert 0 <= mean <= 1
    assert std >= 0
    # With dropout, std should likely be > 0 unless weights are zero or dropout is 0
    # Our model has dropout=0.3 by default.
    
    # Check shape of forward pass
    model = trainer.model
    model.eval()
    with torch.no_grad():
        out = model(sample_input)
    assert out.shape == (1, 1)

def test_compute_features(trainer):
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="min"),
        "pcr": np.linspace(0, 1, 10), # Trend = 0.111
        "gex": np.zeros(10),
        "max_pain": np.zeros(10),
        "iv_skew": np.zeros(10)
    })
    
    feats = trainer.compute_derived_features(df)
    assert 'pcr_trend' in feats.columns
    assert len(feats) == 10
    # Standardized, so won't be exactly diff

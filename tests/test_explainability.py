import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import MagicMock, patch

# Mock shap module before importing explainability
sys.modules['shap'] = MagicMock()

# Now import
from explainability import Explainer

def test_shap_computation_mock ():
    # Force SHAP_AVAILABLE to True for test logic check (or mock the internal check if possible)
    # Actually, if we mocked the module, the try-except in explainability.py might catch it as available or not depending on how we mocked.
    # But since I added a try-except block in explainability.py, if the import succeeds (which it will with MagicMock), SHAP_AVAILABLE is True.
    
    ex = Explainer()
    
    # Mock X
    X = pd.DataFrame(np.random.rand(10, 3), columns=['f1', 'f2', 'f3'])
    model = MagicMock()
    
    # Mock shap.TreeExplainer
    with patch('explainability.shap') as mock_shap:
        mock_explainer = MagicMock()
        mock_shap.TreeExplainer.return_value = mock_explainer
        
        # Mock shap_values return
        # Shape should represent (n_samples, n_features)
        # We pass 1 sample (last row) -> output should be (1, 3)
        mock_explainer.shap_values.return_value = np.array([[0.1, -0.2, 0.5]])
        
        result = ex.compute_shap(model, X, top_k=3)
        
        # Verify structure
        assert "shap_values" in result
        assert "top_features" in result
        assert result['top_features']['f3'] == 0.5 # Largest abs value
        assert len(result['top_features']) == 3

def test_report_generation():
    ex = Explainer()
    logs = [
        {'top_features': {'f1': 0.5, 'f2': 0.1}},
        {'top_features': {'f1': 0.4, 'f3': 0.2}},
        {'top_features': {'f2': 0.3, 'f1': 0.1}}
    ]
    
    output = "test_report.html"
    ex.generate_daily_report(logs, output_path=output)
    
    assert os.path.exists(output)
    
    if os.path.exists(output):
        os.remove(output)

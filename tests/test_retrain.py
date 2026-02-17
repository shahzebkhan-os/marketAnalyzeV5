import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, AsyncMock, patch
from retrain import ModelScheduler

@pytest.mark.asyncio
async def test_scheduler_promotion_logic():
    # Mock DB Pool
    mock_pool = MagicMock()
    
    scheduler = ModelScheduler(mock_pool)
    
    # Mock Trainer
    scheduler.trainer = MagicMock()
    scheduler.trainer.train_models.return_value = {
        "lgbm": {"auc": 0.70} # High AUC
    }
    
    # Mock DB Interaction for 'active_auc'
    # In the simplified code, it was hardcoded 0.55.
    # In a real test, we'd mock the query result.
    
    # Since we are mocking register_model in the module import, we need to patch it.
    with patch("retrain.register_model", new_callable=AsyncMock) as mock_register:
         # Run
         await scheduler.run_retraining()
         
         # Should call register_model with SHADOW because 0.70 > 0.55
         mock_register.assert_called_once()
         args, kwargs = mock_register.call_args
         assert kwargs['status'] == 'SHADOW'
         assert kwargs['metrics']['auc'] == 0.70

@pytest.mark.asyncio
async def test_scheduler_no_promotion():
    mock_pool = MagicMock()
    scheduler = ModelScheduler(mock_pool)
    
    # Candidate worse than 0.55
    scheduler.trainer = MagicMock()
    scheduler.trainer.train_models.return_value = {
        "lgbm": {"auc": 0.50} 
    }
    
    with patch("retrain.register_model", new_callable=AsyncMock) as mock_register:
         await scheduler.run_retraining()
         
         # Should NOT call register_model
         mock_register.assert_not_called()

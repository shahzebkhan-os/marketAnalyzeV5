import os
import pickle
import numpy as np
import pandas as pd
import logging
import lightgbm as lgb
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class StackingEnsemble:
    def __init__(self, models_dir: str = "models", penalty_factor: float = 0.5):
        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, "stacking_lgbm.pkl")
        self.stacker = None
        self.penalty_factor = penalty_factor
        
        # Ensure models dir exists
        os.makedirs(models_dir, exist_ok=True)
        self.load_model()

    def train_stacker(self, X_meta: pd.DataFrame, y: pd.Series):
        """
        Trains the meta-learner (LightGBM Regressor).
        X_meta should contain:
        - base_model_probs (lgbm, xgb, logreg, lstm)
        - regime_label (categorical or encoded)
        - variances (lstm_var)
        """
        logger.info("Training Stacking Ensemble...")
        
        # Initialize LGBM Regressor
        # We use regression to predict the probability (0-1) directly, optimizing MSE or BinaryLogLoss
        # standard LGBMClassifier is also an option, but Regressor gives us a continuous score easier 
        # without sigmoid mapping if we just want "confidence".
        # Let's use Regressor with output limited to [0,1] conceptually.
        
        self.stacker = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            n_estimators=100,
            learning_rate=0.05,
            random_state=42
        )
        
        self.stacker.fit(X_meta, y)
        self.save_model()
        logger.info("Stacking ensemble trained and saved.")

    def predict(self, base_preds: Dict[str, float], regime_info: Dict[str, Any], variance: float) -> Dict[str, Any]:
        """
        Predicts final confidence score.
        Args:
            base_preds: Dict of {'lgbm': 0.6, 'xgb': 0.55, 'lstm': 0.7}
            regime_info: Dict with 'label' or 'hmm_state' etc.
            variance: Variance from MC Dropout (Sequence Model).
        Returns:
            Dict containing final_confidence (0-100) and breakdown.
        """
        if self.stacker is None:
            logger.warning("Stacker not trained. returning average of base preds.")
            avg_prob = np.mean(list(base_preds.values()))
            return {"final_confidence": avg_prob * 100, "reason": "Average (No Stacker)"}

        # Construct feature vector
        # Must match training time columns. 
        # For simplicity, we assume fixed order: [lgbm, xgb, logreg, lstm, regime_val, variance]
        # In a real system, we'd ensure column alignment using a transformer or named columns.
        
        # We need to encode regime. 
        # regime_info['label']: "Trending", "Ranging", "Explosive"
        regime_map = {"Ranging": 0, "Trending": 1, "Explosive": 2}
        regime_val = regime_map.get(regime_info.get("label", "Ranging"), 0)
        
        # Feature vector construction
        # Note: Ensure these keys match what was used in training! 
        # For this implementation, we define the standard schema here.
        
        # Default order: lgbm, xgb, logreg, lstm, regime, variance
        features = [
            base_preds.get('lgbm', 0.5),
            base_preds.get('xgb', 0.5),
            base_preds.get('logreg', 0.5),
            base_preds.get('lstm', 0.5),
            regime_val,
            variance
        ]
        
        X_in = np.array([features])
        
        # 1. Meta-Prediction
        raw_prob = self.stacker.predict(X_in)[0]
        
        # Clip to [0, 1]
        raw_prob = np.clip(raw_prob, 0.0, 1.0)
        
        # 2. Penalize Variance
        # Formula: Confidence = Prob * (1 - penalty * normalized_variance)
        # We assume variance is small (e.g. < 0.25). 
        # penalty_factor defaults to 0.5.
        
        penalty = self.penalty_factor * variance
        final_prob = raw_prob - penalty
        final_prob = np.clip(final_prob, 0.0, 1.0)
        
        final_conf = final_prob * 100.0
        
        breakdown = {
            "base_probs": base_preds,
            "meta_prob": float(raw_prob),
            "variance": variance,
            "penalty": float(penalty),
            "final_confidence": float(final_conf),
            "regime": regime_info.get("label")
        }
        
        logger.info(f"Ensemble Prediction: {breakdown}")
        
        return breakdown

    def save_model(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.stacker, f)

    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.stacker = pickle.load(f)

import os
import json
import pickle
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator
import lightgbm as lgb
import xgboost as xgb

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, registry_path: str = "models/registry.json", models_dir: str = "models"):
        self.registry_path = registry_path
        self.models_dir = models_dir
        self.seed = 42
        self._ensure_dirs()

    def _ensure_dirs(self):
        os.makedirs(self.models_dir, exist_ok=True)
        if not os.path.exists(self.registry_path):
            with open(self.registry_path, 'w') as f:
                json.dump({}, f)

    def set_seed(self, seed: int):
        self.seed = seed
        np.random.seed(seed)

    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Trains LGBM, XGB, and Logistic Regression models.
        Returns a dictionary of trained models and their CV scores.
        """
        self.set_seed(self.seed)
        
        models = {
            "lgbm": lgb.LGBMClassifier(random_state=self.seed, verbose=-1),
            "xgb": xgb.XGBClassifier(random_state=self.seed, verbosity=0, use_label_encoder=False),
            "logreg": LogisticRegression(random_state=self.seed, solver='liblinear')
        }
        
        results = {}
        
        # TimeSeriesSplit for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            cv_scores = []
            
            # Cross-validation
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                try:
                    score = roc_auc_score(y_val, y_pred_proba)
                    cv_scores.append(score)
                except ValueError:
                    # Handle cases with only one class in validation set
                    logger.warning(f"Skipping fold for {name} due to single class in y_val")
            
            avg_auc = np.mean(cv_scores) if cv_scores else 0.0
            logger.info(f"{name} CV AUC: {avg_auc:.4f}")
            
            # Train on full dataset (base model for feature importance)
            model.fit(X, y)
            importance = self.get_feature_importance(model, list(X.columns))
            
            # Calibrate (Isotonic) using CV
            # This creates an ensemble of calibrated classifiers
            final_calibrated = CalibratedClassifierCV(models[name].__class__(**models[name].get_params()), method='isotonic', cv=tscv)
            final_calibrated.fit(X, y)
             
            results[name] = {
                "model": final_calibrated,
                "auc": avg_auc,
                "feature_names": list(X.columns),
                "importance": importance
            }
            
        return results

    def save_model(self, name: str, model_data: Dict[str, Any], metadata: Dict[str, Any] = None):
        """
        Saves the model to disk and updates the registry.
        """
        timestamp = datetime.now().isoformat()
        filename = f"{name}_{timestamp.replace(':', '-')}.pkl"
        filepath = os.path.join(self.models_dir, filename)
        
        # Save pickle
        with open(filepath, 'wb') as f:
            pickle.dump(model_data["model"], f)
            
        # Update registry
        entry = {
            "filepath": filepath,
            "auc": model_data["auc"],
            "features": model_data["feature_names"],
            "importance": model_data.get("importance", {}),
            "timestamp": timestamp,
            "seed": self.seed,
            "metadata": metadata or {}
        }
        
        # Load registry
        with open(self.registry_path, 'r') as f:
            try:
                registry = json.load(f)
            except json.JSONDecodeError:
                registry = {}
            
        if name not in registry:
            registry[name] = []
        registry[name].append(entry)
        
        # Save registry
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=4)
            
        logger.info(f"Saved model {name} to {filepath}")

    def get_feature_importance(self, model: BaseEstimator, feature_names: List[str]) -> Dict[str, float]:
        """
        Extracts feature importance from a base estimator.
        """
        try:
            if hasattr(model, 'feature_importances_'):
                return dict(zip(feature_names, model.feature_importances_.tolist()))
            elif hasattr(model, 'coef_'):
                return dict(zip(feature_names, model.coef_[0].tolist()))
            else:
                return {}
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return {}

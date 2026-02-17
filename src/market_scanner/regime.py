import numpy as np
import pandas as pd
import pickle
import os
import logging
from typing import Dict, Any, Tuple
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class RegimeDetector:
    def __init__(self, models_dir: str = "models", history_window: int = 100):
        self.models_dir = models_dir
        self.history_window = history_window
        self.scaler = StandardScaler()
        self.hmm_model = None
        self.kmeans_model = None
        
        # Ensure models dir exists
        os.makedirs(models_dir, exist_ok=True)
        
        # Paths
        self.hmm_path = os.path.join(models_dir, "hmm_regime.pkl")
        self.kmeans_path = os.path.join(models_dir, "kmeans_iv.pkl")
        self.load_models()

    def load_models(self):
        """Loads existing models if they exist."""
        if os.path.exists(self.hmm_path):
            with open(self.hmm_path, 'rb') as f:
                saved_data = pickle.load(f)
                # Check if it's the old format (just model) or new (dict with scaler)
                if isinstance(saved_data, dict) and 'hmm' in saved_data:
                    self.hmm_model = saved_data['hmm']
                    self.scaler = saved_data['scaler']
                else:
                    self.hmm_model = saved_data
        
        if os.path.exists(self.kmeans_path):
            with open(self.kmeans_path, 'rb') as f:
                self.kmeans_model = pickle.load(f)

    def save_models(self):
        """Saves models to disk."""
        if self.hmm_model:
            with open(self.hmm_path, 'wb') as f:
                # Save both model and scaler
                pickle.dump({'hmm': self.hmm_model, 'scaler': self.scaler}, f)
        
        if self.kmeans_model:
            with open(self.kmeans_path, 'wb') as f:
                pickle.dump(self.kmeans_model, f)

    def fit_models(self, df: pd.DataFrame):
        """
        Fits HMM and K-Means on historical data.
        Expected columns: 'returns', 'volatility', 'iv'
        """
        if len(df) < 50:
            logger.warning("Not enough data to fit regime models")
            return

        # Calculate Returns if not present
        if 'returns' not in df.columns and 'close' in df.columns:
            df['returns'] = df['close'].pct_change().fillna(0)
            
        # Calculate Volatility proxy if not present (using ATR/Close)
        if 'volatility' not in df.columns and 'atr' in df.columns and 'close' in df.columns:
            df['volatility'] = df['atr'] / df['close']

        if 'returns' not in df.columns or 'volatility' not in df.columns:
             logger.warning("Missing required columns for HMM (returns, volatility)")
             return

        X_hmm = df[['returns', 'volatility']].dropna().values
        
        # Fit scaler
        self.scaler.fit(X_hmm)
        X_scaled = self.scaler.transform(X_hmm)
        
        if len(X_scaled) > 0:
            try:
                self.hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
                self.hmm_model.fit(X_scaled)
                logger.info("Fitted HMM model")
            except Exception as e:
                logger.error(f"Failed to fit HMM: {e}")

        # Prepare K-Means data (IV)
        X_iv = df[['iv']].dropna().values
        if len(X_iv) > 0:
            self.kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
            self.kmeans_model.fit(X_iv)
            logger.info("Fitted K-Means model")
            
        self.save_models()
        X_iv = df[['iv']].dropna().values
        if len(X_iv) > 0:
            self.kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
            self.kmeans_model.fit(X_iv)
            logger.info("Fitted K-Means model")
            
        self.save_models()

    def detect_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detects the current regime based on the latest data point and history.
        Returns a dict with label and details.
        """
        if df.empty:
            return {"label": "Insufficient Data", "details": {}}
        
        # Ensure we have necessary features
        required_cols = ['close', 'atr', 'iv']
        if not all(col in df.columns for col in required_cols):
            return {"label": "Missing Features", "details": {}}

        # Calculate Returns if not present
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change().fillna(0)
            
        # Calculate Volatility proxy if not present (using ATR/Close)
        if 'volatility' not in df.columns:
            df['volatility'] = df['atr'] / df['close']

        # If models are not trained and we have enough history, fit them
        if (self.hmm_model is None or self.kmeans_model is None) and len(df) > 50:
            self.fit_models(df)

        current_state = {}
        
        # 1. HMM Detection
        hmm_state = -1
        if self.hmm_model:
            X_curr = df[['returns', 'volatility']].iloc[-1:].values
            try:
                X_scaled = self.scaler.transform(X_curr)
                hmm_state = self.hmm_model.predict(X_scaled)[0]
            except Exception as e:
                # logger.warning("HMM prediction failed", error=str(e))
                pass
        current_state['hmm_state'] = int(hmm_state)

        # 2. IV Cluster
        iv_cluster = -1
        if self.kmeans_model:
            iv_curr = df[['iv']].iloc[-1:].values
            try:
                iv_cluster = self.kmeans_model.predict(iv_curr)[0]
            except:
                pass
        current_state['iv_cluster'] = int(iv_cluster)

        # 3. ATR Percentile
        atr_curr = df['atr'].iloc[-1]
        atr_hist = df['atr'].iloc[-self.history_window:]
        atr_pct = (atr_hist < atr_curr).mean()
        current_state['atr_pct'] = float(atr_pct)

        # Ensemble Logic for Label
        # Heuristic: 
        # High ATR Pct (>0.9) -> Explosive
        # Low ATR Pct (<0.3) & Low IV -> Ranging
        # Else -> Trending (simplified)
        
        label = "Trending" # Default
        if atr_pct > 0.9:
            label = "Explosive"
        elif atr_pct < 0.3:
            label = "Ranging"
            
        # Refine with clustering if available (assuming cluster centers are ordered)
        # But we don't know the order of clusters without inspecting centers.
        # For now, stick to ATR heuristic as primary, model outputs as metadata.
        
        return {
            "label": label,
            "details": current_state
        }

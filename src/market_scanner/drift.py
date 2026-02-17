import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
import logging
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from .alerts import AlertManager

logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, alert_manager: AlertManager = None):
        self.alert_manager = alert_manager or AlertManager()
        
        # Baselines
        self.baseline_features: Optional[pd.DataFrame] = None
        
        # Rolling Metrics
        self.rolling_predictions: Deque[Tuple[float, int]] = deque(maxlen=50) # (prob, true_label)
        self.rolling_auc: Deque[float] = deque(maxlen=10) # Store AUC history
        
        # Thresholds
        self.psi_threshold_warning = 0.25
        self.ks_pvalue_alert = 0.01
        self.min_auc_critical = 0.55
        
        # Safety State
        self.is_model_enabled = True
        self.unsafe_windows_count = 0
        
    def set_baseline(self, df: pd.DataFrame):
        """Sets the training baseline for drift comparison."""
        self.baseline_features = df.copy()
        logger.info(f"Drift baseline set: shape={df.shape}")

    def compute_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        """
        def scale_range(input, min_val, max_val):
            input += (1e-6) # Avoid div by zero
            return (input - min_val) / (max_val - min_val)

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
        breakpoints = np.percentile(expected, breakpoints)
        
        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
        
        # Avoid zero division
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return psi

    def check_drift(self, current_batch: pd.DataFrame) -> List[Dict]:
        """
        Checks for drift in the current batch of features vs baseline.
        Returns list of drift logs.
        """
        if self.baseline_features is None:
            return []
            
        logs = []
        
        # Only check numeric columns intersection
        cols = self.baseline_features.select_dtypes(include=np.number).columns.intersection(current_batch.columns)
        
        for col in cols:
             # Sample for speed if large
             base_vals = self.baseline_features[col].values
             curr_vals = current_batch[col].values
             
             # KS Test
             ks_stat, p_value = ks_2samp(base_vals, curr_vals)
             
             if p_value < self.ks_pvalue_alert:
                 status = "CRITICAL"
                 self.alert_manager.send_message(f"🚨 DRIFT ALERT: KS Test failed for {col} (p={p_value:.5f})", priority="CRITICAL")
                 logs.append({
                     "feature": col, "metric": "KS_PVAL", "value": p_value, "threshold": self.ks_pvalue_alert, "status": status
                 })

             # PSI Test
             try:
                psi = self.compute_psi(base_vals, curr_vals)
                if psi > self.psi_threshold_warning:
                    status = "WARNING"
                    # self.alert_manager.send_message(f"⚠️ DRIFT WARNING: High PSI for {col} ({psi:.2f})", priority="WARNING")
                    logs.append({
                        "feature": col, "metric": "PSI", "value": psi, "threshold": self.psi_threshold_warning, "status": status
                    })
             except Exception as e:
                 logger.warning(f"PSI computation failed for {col}: {e}")
                 
        return logs

    def update_rolling_auc(self, y_true: int, y_prob: float):
        """
        Updates rolling Window and calculates AUC.
        Checks safety.
        """
        self.rolling_predictions.append((y_prob, y_true))
        
        if len(self.rolling_predictions) >= 20: # Min samples to calc AUC
            try:
                probs, labels = zip(*self.rolling_predictions)
                if len(set(labels)) > 1: # Need both classes
                    auc = roc_auc_score(labels, probs)
                    self.rolling_auc.append(auc)
                    self._check_safety(auc)
                else:
                    # Not enough class diversity yet
                    pass 
            except Exception as e:
                logger.error(f"Rolling AUC calc failed: {e}")
    
    def _check_safety(self, current_auc: float):
        """
        Disables model if AUC is critical for consecutive windows.
        """
        if current_auc < self.min_auc_critical:
            self.unsafe_windows_count += 1
        else:
            self.unsafe_windows_count = 0
            
        if self.unsafe_windows_count >= 3:
            if self.is_model_enabled:
                self.is_model_enabled = False
                msg = f"⛔ SAFETY STOP: Rolling AUC < {self.min_auc_critical} for 3 windows. Model DISABLED."
                logger.critical(msg)
                self.alert_manager.send_message(msg, priority="CRITICAL")
        elif not self.is_model_enabled and self.unsafe_windows_count == 0 and current_auc > 0.6:
            # Auto-recovery logic could go here, or manual reset
            # For now, require manual intervention or restart, but let's log potential recovery
            logger.info(f"Safety condition improving: auc={current_auc}")

    def monitor_prediction_distribution(self, probs: np.ndarray):
        """
        Checks for overconfidence or collapse.
        """
        mean_prob = np.mean(probs)
        
        # Check collapse to 0.5
        if 0.48 < mean_prob < 0.52 and np.std(probs) < 0.01:
             logger.warning("Prediction collapse detected (all ~0.5)")
             
        # Check extreme overconfidence
        high_conf = np.mean(probs > 0.95)
        if high_conf > 0.5:
             logger.warning("Extreme overconfidence detected (>50% of predictions > 0.95)")

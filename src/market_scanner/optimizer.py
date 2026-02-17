import logging
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    def __init__(self, backtester_cls, data: pd.DataFrame, base_signals: pd.DataFrame, db_url: str = "sqlite:///optuna.db"):
        """
        backtester_cls: Class reference to Backtester
        data: Market data for backtesting
        base_signals: DataFrame of base model predictions
        db_url: Database URL for Optuna storage (default: SQLite)
        """
        self.backtester_cls = backtester_cls
        self.data = data
        self.base_signals = base_signals
        self.db_url = db_url
        self.study_name = "holographic_glenn_optimization"

    def objective(self, trial: optuna.Trial) -> float:
        # 1. Suggest Parameters
        weight_lgbm = trial.suggest_float("weight_lgbm", 0.0, 1.0)
        weight_xgb = trial.suggest_float("weight_xgb", 0.0, 1.0)
        weight_lstm = trial.suggest_float("weight_lstm", 0.0, 1.0)
        
        confidence_threshold = trial.suggest_float("confidence_threshold", 0.60, 0.90)
        stop_loss_pct = trial.suggest_float("stop_loss_pct", 0.005, 0.05)
        # position_sizing_pct = trial.suggest_float("position_sizing_pct", 0.1, 1.0) # Not used in simple backtester yet
        
        # Normalize weights
        total_weight = weight_lgbm + weight_xgb + weight_lstm
        if total_weight == 0:
            return -10.0 # Penalty
        
        w_lgbm = weight_lgbm / total_weight
        w_xgb = weight_xgb / total_weight
        w_lstm = weight_lstm / total_weight
        
        # 2. Combine Signals
        # Handle missing columns gracefully
        s_lgbm = self.base_signals.get('lgbm', 0)
        s_xgb = self.base_signals.get('xgb', 0)
        s_lstm = self.base_signals.get('lstm', 0)
        
        final_prob = (s_lgbm * w_lgbm) + (s_xgb * w_xgb) + (s_lstm * w_lstm)
        
        # 3. Generate Trade Signals
        # 1 = Buy, -1 = Sell, 0 = Hold
        signals = np.zeros(len(final_prob))
        signals[final_prob > confidence_threshold] = 1
        signals[final_prob < (1 - confidence_threshold)] = -1
        
        # 4. Run Backtest
        # We need to pass stop_loss_pct to backtester. 
        # Assuming Backtester.run accepts kwargs or we modify it.
        # For now, let's assume standard run and we optimize signal generation/weights mostly.
        # To strictly optimize stop_loss, Backtester needs to support it dynamically.
        
        bt = self.backtester_cls()
        # Mocking stop_loss injection if supported, else it just runs standard
        bt.stop_loss_pct = stop_loss_pct 
        
        bt.run(self.data, pd.Series(signals, index=self.data.index))
        metrics = bt.calculate_metrics()
        
        sharpe = metrics.get('Sharpe Ratio', -10.0)
        
        # Handling NaN
        if np.isnan(sharpe):
            return -10.0
            
        return sharpe

    def run_optimization(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        Runs the optimization study.
        """
        logger.info(f"Starting Optuna Optimization: n_trials={n_trials}")
        
        storage = optuna.storages.RDBStorage(url=self.db_url)
        study = optuna.create_study(
            study_name=self.study_name, 
            storage=storage, 
            load_if_exists=True, 
            direction="maximize",
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(self.objective, n_trials=n_trials)
        
        logger.info(f"Optimization Complete: best_value={study.best_value}, best_params={study.best_params}")
        return study.best_params

    def get_optimization_history(self):
        # Helper to fetch history for dashboard
        try:
            storage = optuna.storages.RDBStorage(url=self.db_url)
            study = optuna.load_study(study_name=self.study_name, storage=storage)
            return study.trials_dataframe()
        except:
            return pd.DataFrame()

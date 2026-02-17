import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from .portfolio import Portfolio, Position

class RiskEngine:
    def __init__(self):
        # Thresholds
        self.MAX_CAPITAL_UTILIZATION = 0.05 # 5%
        self.DAILY_LOSS_LIMIT_PCT = 0.02 # 2%
        self.DELTA_THRESHOLD = 500.0
        self.GAMMA_THRESHOLD = 1000.0
    
    def calculate_portfolio_variance(self, portfolio: Portfolio, lookback_days: int = 90) -> float:
        """
        Calculates portfolio variance using w^T * Sigma * w.
        Uses mock covariance matrix for NIFTY, BANKNIFTY, SENSEX.
        """
        symbols = sorted(list(set(p.symbol for p in portfolio.positions)))
        if not symbols: return 0.0
        
        # Mock Covariance Matrix (Daily Returns)
        # NIFTY, BANKNIFTY, SENSEX are highly correlated (>0.9)
        # Variance roughly 1% daily -> 0.0001
        n = len(symbols)
        sigma = np.ones((n, n)) * 0.00009 # High internal correlation
        np.fill_diagonal(sigma, 0.0001)   # Variance on diagonal
        
        # Weight Vector (Net Exposure per symbol)
        # Normalize by Capital? Or just nominal variance?
        # Usually VaR is calculated on nominal exposure.
        weights = []
        for sym in symbols:
            exposure = sum(p.market_value for p in portfolio.positions if p.symbol == sym)
            weights.append(exposure)
            
        w = np.array(weights)
        
        # P_Var = w^T * Sigma * w (in currency units squared)
        portfolio_var = w.T @ sigma @ w
        return float(portfolio_var)

    def check_exposure_limits(self, portfolio: Portfolio, current_daily_pnl: float) -> List[str]:
        """
        Returns a list of limit breach warnings.
        """
        warnings = []
        
        # 1. Capital Utilization
        utilization = portfolio.get_capital_utilization()
        if utilization > self.MAX_CAPITAL_UTILIZATION:
            warnings.append(f"Capital Utilization {utilization:.1%} > {self.MAX_CAPITAL_UTILIZATION:.1%}")
            
        # 2. Daily Loss Limit
        loss_pct = -current_daily_pnl / portfolio.initial_capital
        if loss_pct > self.DAILY_LOSS_LIMIT_PCT:
            warnings.append(f"Daily Loss {loss_pct:.1%} > {self.DAILY_LOSS_LIMIT_PCT:.1%}")
            
        return warnings

    def propose_hedge(self, net_greeks: Dict[str, float]) -> List[str]:
        """
        Generates hedge suggestions based on Net Greeks.
        """
        suggestions = []
        
        # Delta Hedge
        net_delta = net_greeks.get("delta", 0.0)
        if net_delta > self.DELTA_THRESHOLD:
            qty_to_sell = int(abs(net_delta)) # Assuming Delta 1 for Futures
            suggestions.append(f"SELL {qty_to_sell} NIFTY FUT (Delta Hedge)")
        elif net_delta < -self.DELTA_THRESHOLD:
            qty_to_buy = int(abs(net_delta))
            suggestions.append(f"BUY {qty_to_buy} NIFTY FUT (Delta Hedge)")
            
        # Gamma Hedge
        net_gamma = net_greeks.get("gamma", 0.0)
        if abs(net_gamma) > self.GAMMA_THRESHOLD:
            # High Gamma = High Risk of large moves hurting (if short gamma) or helping (if long).
            # If Net Gamma is very negative (likely short options), suggest Long Straddle.
            if net_gamma < 0:
                suggestions.append("BUY ATM STRADDLE (Gamma Protection)")
        
        return suggestions

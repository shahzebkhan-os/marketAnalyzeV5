import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from portfolio import Portfolio, Position
from scipy.stats import norm

class MonteCarloEngine:
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def simulate_gbm(self, S0: float, mu: float, sigma: float, T: float, n_sims: int, n_steps: int) -> np.ndarray:
        """
        Simulates Geometric Brownian Motion paths.
        Returns array of shape (n_sims, n_steps + 1).
        """
        dt = T / n_steps
        # GBM: S_t = S_0 * exp((mu - 0.5*sigma^2)*t + sigma*W_t)
        
        # Increments
        # dW ~ N(0, sqrt(dt))
        dW = np.random.normal(0, np.sqrt(dt), size=(n_sims, n_steps))
        
        # Cumulative Sum for Brownian Path
        W = np.cumsum(dW, axis=1)
        
        # Time steps
        t = np.linspace(dt, T, n_steps)
        
        # Drift and Diffusion
        # formula: S_t = S0 * exp( (mu - 0.5 * sigma**2) * t + sigma * W )
        
        exponent = (mu - 0.5 * sigma**2) * t + sigma * W
        paths = S0 * np.exp(exponent)
        
        # Prepend S0
        start_prices = np.full((n_sims, 1), S0)
        return np.hstack((start_prices, paths))

    def price_option_bs(self, S: np.ndarray, K: float, T: float, r: float, sigma: float, option_type: str) -> np.ndarray:
        """
        Black-Scholes pricing for arrays of spot prices.
        """
        if T <= 0:
            if option_type == 'CE':
                return np.maximum(S - K, 0)
            else:
                return np.maximum(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'CE':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
        return price

    def run_portfolio_simulation(self, portfolio: Portfolio, n_sims: int = 2000, days_horizon: int = 1) -> Dict[str, float]:
        """
        Simulates portfolio PnL distribution over 'days_horizon'.
        """
        np.random.seed(self.seed)
        
        # Assumptions for simulation
        mu = 0.0 # Neutrally drift for short term risk
        r = 0.05 # Risk free rate
        T_horizon = days_horizon / 252.0
        
        total_pnl_sim = np.zeros(n_sims)
        
        # We need to simulate per underlying.
        # Group positions by symbol.
        positions_by_symbol = {}
        for p in portfolio.positions:
            if p.symbol not in positions_by_symbol:
                positions_by_symbol[p.symbol] = []
            positions_by_symbol[p.symbol].append(p)
            
        for symbol, positions in positions_by_symbol.items():
            # Estimate vol from positions if possible, or default.
            # Using mock vol for now.
            sigma = 0.20 # 20% IV
            S0 = positions[0].current_price if positions else 22500 # Fallback
            
            # Simulate underlying path
            # We only need the price at T_horizon
            # S_T = S0 * exp(...)
            # Simple geometric step
            z = np.random.normal(0, 1, n_sims)
            S_T = S0 * np.exp((mu - 0.5 * sigma**2) * T_horizon + sigma * np.sqrt(T_horizon) * z)
            
            for p in positions:
                # Calculate current value
                # Assuming simple pricing logic or stored greeks?
                # For MC, we need full re-pricing to capture non-linearity (Gamma).
                
                # Current Value
                # val_0 = p.market_value # This is (Price - Entry) * Qty? No, Market Value is Price * Qty.
                # Actually, pnl = Val_T - Val_0
                
                # We need Option specifics (Strike, Expiry).
                # If Future, simple linear.
                
                if p.instrument_type == 'FUT':
                    val_T = (S_T - p.entry_price) * p.qty 
                    val_0 = (p.current_price - p.entry_price) * p.qty
                
                elif p.instrument_type in ['CE', 'PE']:
                    # Time to expiry
                    # Mocking T_exp based on string or default
                    T_exp = 30.0/365.0 
                    T_remain = T_exp - T_horizon
                    
                    # Re-price at Horizon
                    prices_T = self.price_option_bs(S_T, p.strike, T_remain, r, sigma, p.instrument_type)
                    val_T = prices_T * p.qty
                    
                    # Price Now
                    price_0 = self.price_option_bs(np.array([p.current_price]), p.strike, T_exp, r, sigma, p.instrument_type)[0]
                    val_0 = price_0 * p.qty
                else:
                    val_T = 0
                    val_0 = 0
                
                pnl_change = val_T - val_0
                total_pnl_sim += pnl_change

        # Compute VaR
        # 95% VaR is the 5th percentile of PnL distribution
        var_95 = np.percentile(total_pnl_sim, 5)
        var_99 = np.percentile(total_pnl_sim, 1)
        worst_case = np.min(total_pnl_sim)
        
        return {
            "var_95": var_95,
            "var_99": var_99,
            "worst_case": worst_case,
            "pnl_sim": total_pnl_sim # Return raw for plotting
        }

    def run_stress_test(self, portfolio: Portfolio) -> Dict[str, float]:
        """
        Runs specific stress scenarios.
        """
        results = {}
        
        # Scenario 1: Gap Up 2%
        pnl_up = 0.0
        for p in portfolio.positions:
            # Simplified Delta-Gamma approx for stress test if pricing not avail?
            # Or full re-price. Let's use Delta-Gamma for speed/simplicity here as we lack full pricing engine context in this file.
            # PnL ~ Delta * dS + 0.5 * Gamma * dS^2
            
            greeks = p.greeks
            delta = greeks.get("delta", 0.0) * p.qty
            gamma = greeks.get("gamma", 0.0) * p.qty
            
            S = p.current_price 
            dS = S * 0.02
            
            pnl = delta * dS + 0.5 * gamma * (dS**2)
            pnl_up += pnl
            
        results["gap_up_2pct"] = pnl_up
        
        # Scenario 2: Gap Down 2%
        pnl_down = 0.0
        for p in portfolio.positions:
            greeks = p.greeks
            delta = greeks.get("delta", 0.0) * p.qty
            gamma = greeks.get("gamma", 0.0) * p.qty
            
            S = p.current_price 
            dS = S * -0.02
            
            pnl = delta * dS + 0.5 * gamma * (dS**2)
            pnl_down += pnl
            
        results["gap_down_2pct"] = pnl_down
        
        # Scenario 3: IV Spike +20% (Relative)
        pnl_iv = 0.0
        for p in portfolio.positions:
            greeks = p.greeks
            vega = greeks.get("vega", 0.0) * p.qty
            
            # Vega is change in PnL per 1% change in vol.
            # If IV is 20%, +20% relative is +4% absolute (20 * 1.2 = 24).
            # Assuming IV ~ 20.
            dSigma = 4.0 # +4 vol points
            
            pnl = vega * dSigma
            pnl_iv += pnl
            
        results["iv_spike_20pct"] = pnl_iv
        
        return results

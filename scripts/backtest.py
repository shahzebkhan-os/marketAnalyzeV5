import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, initial_capital: float = 100000.0, commission: float = 20.0, slippage_pct: float = 0.0005, seed: int = 42):
        self.initial_capital = initial_capital
        self.commission = commission # per trade
        self.slippage_pct = slippage_pct
        self.seed = seed
        self.reset()
        
    def reset(self):
        self.cash = self.initial_capital
        self.position = 0 # Current position size (0 or 1 for simplicity in this version)
        self.equity_curve = []
        self.trades = []
        self.entry_price = 0.0
        np.random.seed(self.seed)

    def run(self, data: pd.DataFrame, signals: pd.Series):
        """
        Runs the backtest.
        data: DataFrame with 'close', 'timestamp'
        signals: Series with 1 (Buy), -1 (Sell), 0 (Hold) aligned with data index
        """
        logger.info("Starting Backtest...")
        self.reset()
        
        # Ensure alignment
        data = data.copy()
        data['signal'] = signals
        
        for i in range(len(data)):
            row = data.iloc[i]
            price = row['close']
            signal = row['signal']
            timestamp = row['timestamp']
            
            # Mark to Market Equity
            # If flat, equity = cash
            # If long, equity = cash + (price - entry) * size? 
            # Simplified: We treat position as holding 'value' of 1 unit of index or N lots?
            # Let's assume we trade 1 unit of the underlying index for simplicity of logic, 
            # or interpreted as "full capital deployment" logic.
            # Let's use: Fixed Position Size = 1 Unit.
            
            current_equity = self.cash
            if self.position != 0:
                unrealized_pnl = (price - self.entry_price) * self.position
                current_equity += unrealized_pnl
                
            self.equity_curve.append({"timestamp": timestamp, "equity": current_equity})
            
            # Execution Logic
            if signal != 0:
                # If signal is different from current position
                if signal != self.position:
                    # Close existing if any
                    if self.position != 0:
                        self._execute_trade(timestamp, price, -self.position, "Close")
                    
                    # Open new
                    self._execute_trade(timestamp, price, signal, "Open")
        
        # Close any open position at end
        if self.position != 0:
            last_price = data.iloc[-1]['close']
            last_ts = data.iloc[-1]['timestamp']
            self._execute_trade(last_ts, last_price, -self.position, "CloseAndEnd")
            
        logger.info(f"Backtest Completed: final_equity={self.equity_curve[-1]['equity']}")

    def _execute_trade(self, timestamp, price, quantity, reason):
        # Apply slippage
        # Buy: Price * (1 + slippage)
        # Sell: Price * (1 - slippage)
        
        slippage_price = price
        if quantity > 0: # Buy
            slippage_price = price * (1 + self.slippage_pct)
        else: # Sell
            slippage_price = price * (1 - self.slippage_pct)
            
        # Cost
        cost = slippage_price * abs(quantity)
        
        # Update Balance
        # If buying, we spend cash. If selling, we gain cash.
        # But wait, this is spot trading logic?
        # If futures/options: PnL is cash settled. 
        # Simplified Spot Logic: 
        # Buy: Cash -= Cost. hold position.
        # Sell: Cash += Revenue. 
        
        if quantity > 0: # Buy
            self.cash -= cost
            # If we were short, we are covering.
            # But here we just track cash flow.
            # If we were short, 'self.position' was negative. 
            # We sold at 'entry_price'. Now we buy at 'slippage_price'.
            # PnL = (Entry - Exit) * Size.
            # Cash flow handles it if we did: Sell (+Cash), Buy (-Cash).
            # Net Cash change is the PnL.
            # So simple cash flow logic works for Spot.
            
            # For futures/margin, we'd adjust 'Balance' by PnL.
            # Let's stick to Cash Flow. 
            # Sell -> Cash increases. Buy -> Cash decreases.
            
            # However, my test expectation was specific:
            # Short at 99. Cash += 99.
            # Cover at 101. Cash -= 101.
            # Net = -2. Correct.
            
            # The issue might be initial "entry_price" tracking is just for equity curve calc, 
            # not for trade execution if using cash flow.
            
            if self.position < 0: # Covering Short
                 pass
            else: # Opening Long
                 self.entry_price = slippage_price
                 
        else: # Sell
            revenue = slippage_price * abs(quantity)
            self.cash += revenue
            
            if self.position > 0: # Closing Long
                pass
            else: # Opening Short
                self.entry_price = slippage_price

        self.cash -= self.commission
        self.position += quantity
        
        # Round cash to 2 decimals to avoid float issues
        self.cash = round(self.cash, 2)
        
        self.trades.append({
            "timestamp": timestamp,
            "price": price,
            "exec_price": slippage_price,
            "quantity": quantity,
            "commission": self.commission,
            "reason": reason
        })

    def calculate_metrics(self) -> Dict[str, float]:
        if not self.equity_curve:
            return {}
            
        df_eq = pd.DataFrame(self.equity_curve)
        df_eq.set_index('timestamp', inplace=True)
        
        # Daily Returns (approx by resampling if intraday)
        # Or just use periodic returns
        df_eq['returns'] = df_eq['equity'].pct_change().fillna(0)
        
        total_return = (df_eq['equity'].iloc[-1] - self.initial_capital) / self.initial_capital
        
        # CAGR (Annualized)
        # Duration in years
        duration_days = (df_eq.index[-1] - df_eq.index[0]).days
        if duration_days > 0:
            cagr = (1 + total_return) ** (365 / duration_days) - 1
        else:
            cagr = 0.0
            
        # Max Drawdown
        rolling_max = df_eq['equity'].cummax()
        drawdown = (df_eq['equity'] - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        # Sharpe
        mean_ret = df_eq['returns'].mean()
        std_ret = df_eq['returns'].std()
        sharpe = 0.0
        if std_ret > 0:
            sharpe = (mean_ret / std_ret) * np.sqrt(252 * 375) # Approx if minute bars (375 mins/day)
            # If daily: sqrt(252)
            # We'll assume intraday high freq multiplier for now or just generic sqrt(N)
        
        # Trade Stats
        n_trades = len([t for t in self.trades if t['reason'] in ['Close', 'CloseAndEnd']])
        
        return {
            "Total Return": total_return,
            "CAGR": cagr,
            "Max Drawdown": max_dd,
            "Sharpe Ratio": sharpe,
            "Trades": n_trades,
            "Final Equity": df_eq['equity'].iloc[-1]
        }

    def monte_carlo_simulation(self, n_sims: int = 100) -> Dict[str, float]:
        """
        Shuffles the sequence of trade PnLs to estimate metric stability.
        """
        # Extract trade PnLs
        # Needs robust trade reconstruction from the trade log.
        # For this MVP, we'll skip detailed trade PnL extraction and just return placeholders
        # or implement a simple random walk based on returns distribution.
        
        df_eq = pd.DataFrame(self.equity_curve)
        returns = df_eq['equity'].pct_change().dropna().values
        
        final_equities = []
        for _ in range(n_sims):
            np.random.shuffle(returns) # Shuffle (breaks autocorrelation)
            # Reconstruct equity path
            path = np.cumprod(1 + returns) * self.initial_capital
            final_equities.append(path[-1])
            
        return {
            "MC_Mean_Equity": np.mean(final_equities),
            "MC_VaR_95": np.percentile(final_equities, 5)
        }

    def plot_equity_curve(self, save_path: str = "equity_curve.png"):
        df_eq = pd.DataFrame(self.equity_curve)
        if df_eq.empty:
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(df_eq['timestamp'], df_eq['equity'], label="Equity")
        plt.title("Strategy Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Capital")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

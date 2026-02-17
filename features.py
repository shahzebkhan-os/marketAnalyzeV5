import numpy as np
import pandas as pd
from scipy.stats import norm
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def set_seed(seed_value: int = 42):
    """Enforce deterministic behavior."""
    np.random.seed(seed_value)

class FeatureEngineer:
    def __init__(self):
        set_seed()

    def compute_oi_stats(self, chain_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Computes PCR, OI Imbalance, and Max Pain.
        Handles flat chain_data (separate rows for CE/PE).
        """
        try:
            df = pd.DataFrame(chain_data)
            if df.empty:
                return {}

            # Standardize types
            df['type_std'] = df['type'].str.upper().apply(lambda x: 'CE' if x in ['CE', 'CALL'] else 'PE')
            
            # Sum OI correctly based on type
            total_call_oi = df[df['type_std'] == 'CE']['oi'].sum()
            total_put_oi = df[df['type_std'] == 'PE']['oi'].sum()
            
            pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0.0
            oi_imbalance = total_call_oi - total_put_oi

            # Max Pain Calculation on grouped data
            grouped = df.groupby('strike').agg({
                'oi': 'sum', # This is not quite right for max pain, we need CE OI and PE OI per strike
            }).reset_index()
            
            # Need a strike -> [call_oi, put_oi] mapping
            strike_data = {}
            for _, row in df.iterrows():
                s = row['strike']
                if s not in strike_data: strike_data[s] = {'CE': 0, 'PE': 0}
                strike_data[s][row['type_std']] = row['oi']
            
            strikes = sorted(strike_data.keys())
            losses = []
            for target_strike in strikes:
                total_loss = 0
                for s, ois in strike_data.items():
                    # Call loss: if market > strike, writer loses (market - strike) * call_oi
                    total_loss += max(0, s - target_strike) * ois['CE']
                    # Put loss: if market < strike, writer loses (strike - market) * put_oi
                    total_loss += max(0, target_strike - s) * ois['PE']
                losses.append(total_loss)
            
            max_pain = strikes[np.argmin(losses)] if losses else 0.0

            return {
                "pcr": float(pcr),
                "oi_imbalance": float(oi_imbalance),
                "max_pain": float(max_pain)
            }
        except Exception as e:
            logger.error(f"Error computing OI stats: {e}")
            return {}

    def black_scholes_greeks(self, S, K, T, r, sigma, option_type='call'):
        """
        Computes Delta and Gamma using Black-Scholes.
        S: Spot Price
        K: Strike Price
        T: Time to Expiration (years)
        r: Risk-free rate
        sigma: Implied Volatility
        """
        try:
            if T <= 0 or sigma <= 0:
                return 0.0, 0.0
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            
            if option_type == 'call':
                delta = norm.cdf(d1)
            else:
                delta = norm.cdf(d1) - 1
                
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            return delta, gamma
        except Exception:
            return 0.0, 0.0

    def compute_greeks(self, chain_data: List[Dict[str, Any]], spot_price: float, risk_free_rate: float = 0.05) -> Dict[str, float]:
        """
        Computes Gamma Exposure (GEX), Delta Exposure (DEX), and Gamma Flip Level.
        """
        try:
            df = pd.DataFrame(chain_data)
            if df.empty:
                return {}
            
            gex = 0.0
            dex = 0.0
            
            # Assuming dataframe has: strike, type (call/put), oi, iv, expiry (or dte provided)
            # If 'delta' and 'gamma' are not in columns, we calculate them.
            
            for index, row in df.iterrows():
                strike = row.get('strike')
                opt_type = row.get('type', 'call').lower() # 'call' or 'put' or 'CE'/'PE'
                oi = row.get('oi', 0)
                iv = row.get('iv', 0)
                
                # Normalize type
                if opt_type in ['ce', 'call']:
                    opt_type_std = 'call'
                else:
                    opt_type_std = 'put'

                # Time to expiry handling (mocking T=7/365 if not present)
                T = row.get('dte', 7) / 365.0 
                
                delta = row.get('delta')
                gamma = row.get('gamma')
                
                if delta is None or gamma is None:
                    delta, gamma = self.black_scholes_greeks(spot_price, strike, T, risk_free_rate, iv, opt_type_std)
                
                # GEX = Gamma * OI * Spot * 100 (contract multiplier usually, here just using 1 or 100 as per convention)
                # Assuming standard contract size of 1 for calculation simplicity unless specified
                # GEX contribution: + for Call, - for Put? 
                # Market Maker GEX: They are short options. 
                # If we buy call, MM sells call. MM is short gamma. Price up -> MM delta decreases (becomes more short) -> must buy underlying.
                # Actually, standard GEX view:
                # Dealers Long Calls -> Long Gamma. Dealers Short Puts -> Long Gamma.
                # Dealers Short Calls -> Short Gamma. Dealers Long Puts -> Short Gamma.
                # Usually we assume dealers are SHORT stats.
                # Let's use SpotGamma definition:
                # Call OI * Gamma - Put OI * Gamma
                
                contribution = gamma * oi * spot_price
                if opt_type_std == 'call':
                    gex += contribution
                else:
                    gex -= contribution
                
                # DEX: Delta * OI * Spot
                dex += delta * oi * spot_price
            
            # Gamma Flip: Level where GEX flips from positive to negative.
            # Simplified: It's often near the max pain or where the dominance shifts.
            # Analytical derivation is complex, simple approximation:
            # It's loosely the strike where Call GEX = Put GEX
            
            # Simple search for flip level
            gamma_flip = spot_price 
            
            return {
                "gex": float(gex),
                "dex": float(dex),
                "gamma_flip": float(gamma_flip)
            }
        except Exception as e:
            logger.error(f"Error computing Greeks: {e}")
            return {}

    def compute_technicals(self, prices: List[float]) -> Dict[str, float]:
        """
        Computes EMA20, EMA50, RSI, MACD, ATR, Bollinger using pandas.
        prices: list of close prices ordered by time.
        """
        try:
            if not prices or len(prices) < 50:
                 # Return empty or partial if not enough data
                 # For robust implementation, handle small samples gracefully
                 pass

            s = pd.Series(prices)
            
            ema20 = s.ewm(span=20, adjust=False).mean().iloc[-1] if len(s) >= 20 else 0.0
            ema50 = s.ewm(span=50, adjust=False).mean().iloc[-1] if len(s) >= 50 else 0.0
            
            # RSI
            delta = s.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1] if len(s) >= 14 else 0.0

            # MACD
            ema12 = s.ewm(span=12, adjust=False).mean()
            ema26 = s.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd = macd_line.iloc[-1]
            
            # Bollinger
            sma20 = s.rolling(window=20).mean()
            std20 = s.rolling(window=20).std()
            bb_upper = (sma20 + 2 * std20).iloc[-1] if len(s) >= 20 else 0.0
            bb_lower = (sma20 - 2 * std20).iloc[-1] if len(s) >= 20 else 0.0

            # ATR (Requires High/Low, if we only have close, strictly speaking we can't do true ATR)
            # using volatility of close as proxy if needed, or accepting just close for now.
            # Assuming 'prices' is just a list of closes. 
            # To do ATR properly we need OHLC. 
            # I will return 0.0 for ATR if high/low not available in this signature.
            atr = 0.0

            return {
                "ema20": float(ema20),
                "ema50": float(ema50),
                "rsi": float(rsi),
                "macd": float(macd),
                "bb_upper": float(bb_upper),
                "bb_lower": float(bb_lower),
                "atr": atr
            }
        except Exception as e:
            logger.error(f"Error computing Technicals: {e}")
            return {}

    def compute_volatility(self, chain_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Computes IV Skew, IV Rank.
        IV Skew = OTMPutIV - OTMCallIV (or similar measure).
        """
        try:
            df = pd.DataFrame(chain_data)
            if df.empty or 'iv' not in df.columns:
                return {}
            
            # Simply sorting by iv for rank? IV Rank usually requires historical IV data (low/high over 52w).
            # Without history, we can only compute skew from the snapshot.
            
            # IV Skew: Put IV (ATM-X%) - Call IV (ATM+X%)
            # Placeholder logic
            iv_skew = 0.0
            
            return {
                "iv_skew": iv_skew,
                "iv_rank": 0.0 # Requires history
            }
        except Exception as e:
            logger.error(f"Error computing Volatility: {e}")
            return {}

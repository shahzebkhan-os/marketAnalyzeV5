import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ShockDetector:
    def __init__(self, lookback_window=120):
        # Store last 120 snapshots (approx 20 mins if 10s interval, or just count based)
        # We need short term velocity (60s) and medium term stats (1hr)
        self.history = deque(maxlen=lookback_window)
        
        # Thresholds
        self.PRICE_VELOCITY_THRESHOLD = 0.007 # 0.7%
        self.IV_ZSCORE_THRESHOLD = 2.0
        self.PCR_ZSCORE_THRESHOLD = 3.0
        
    def update(self, symbol: str, current_price: float, current_iv: float, current_pcr: float, timestamp: datetime) -> dict:
        """
        Updates internal state and checks for shock conditions.
        Returns dict with 'triggered': True/False and details.
        """
        
        self.history.append({
            'timestamp': timestamp,
            'price': current_price,
            'iv': current_iv,
            'pcr': current_pcr
        })
        
        if len(self.history) < 10:
            return {'triggered': False}
            
        # 1. Price Velocity Check (Last 60 seconds)
        # Find snapshot ~60s ago
        sixty_sec_ago = timestamp - timedelta(seconds=60)
        
        ref_price = None
        for item in reversed(self.history):
            if item['timestamp'] <= sixty_sec_ago:
                ref_price = item['price']
                break
        
        if not ref_price:
            ref_price = self.history[0]['price'] # Fallback to oldest
            
        price_change_pct = abs((current_price - ref_price) / ref_price)
        
        if price_change_pct > self.PRICE_VELOCITY_THRESHOLD:
            logger.critical(f"SHOCK DETECTED: Price Velocity for {symbol}, change={price_change_pct}")
            return {
                'triggered': True,
                'type': 'PRICE_MOVE',
                'details': {'change_pct': price_change_pct, 'threshold': self.PRICE_VELOCITY_THRESHOLD}
            }
            
        # 2. IV Spike Check (Z-Score)
        iv_values = [x['iv'] for x in self.history]
        iv_mean = np.mean(iv_values)
        iv_std = np.std(iv_values)
        
        if iv_std > 0:
            iv_z = (current_iv - iv_mean) / iv_std
            if iv_z > self.IV_ZSCORE_THRESHOLD:
                logger.critical(f"SHOCK DETECTED: IV Spike for {symbol}, z_score={iv_z}")
                return {
                    'triggered': True,
                    'type': 'IV_SPIKE',
                    'details': {'z_score': iv_z, 'current': current_iv, 'mean': iv_mean}
                }

        # 3. PCR Anomaly (Z-Score)
        pcr_values = [x['pcr'] for x in self.history]
        pcr_mean = np.mean(pcr_values)
        pcr_std = np.std(pcr_values)
        
        if pcr_std > 0:
            pcr_z = abs((current_pcr - pcr_mean) / pcr_std)
            if pcr_z > self.PCR_ZSCORE_THRESHOLD:
                logger.critical(f"SHOCK DETECTED: PCR Anomaly for {symbol}, z_score={pcr_z}")
                return {
                    'triggered': True,
                    'type': 'PCR_ANOMALY',
                    'details': {'z_score': pcr_z, 'current': current_pcr}
                }
                
        return {'triggered': False}

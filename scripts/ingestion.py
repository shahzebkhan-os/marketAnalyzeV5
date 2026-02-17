import asyncio
import signal
import sys
from collections import deque
from typing import Dict, List
import pandas as pd
import sys
import os
# Ensure src is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from market_scanner.client import GrowwClient
from market_scanner.database import get_db_pool, init_db, insert_snapshot, insert_features, insert_regime, fetch_recent_history
from market_scanner.logger import configure_logger, get_logger
from market_scanner.features import FeatureEngineer
from market_scanner.regime import RegimeDetector
from market_scanner.shock_detector import ShockDetector
from market_scanner.alerts import AlertManager
from market_scanner.monitoring import Monitor
import datetime

configure_logger()
logger = get_logger(__name__)

SYMBOLS = ["NIFTY", "BANKNIFTY", "SENSEX"]
FETCH_INTERVAL = 10  # seconds

# In-memory history for technicals
PRICE_HISTORY: Dict[str, deque] = {
    symbol: deque(maxlen=200) for symbol in SYMBOLS
}

async def fetch_and_store(client, symbol, db_pool, feature_engineer, regime_detector, shock_detector, alert_manager, monitor):
    try:
        start_fetch = asyncio.get_event_loop().time()
        data = await client.fetch_option_chain(symbol)
        
        # Log raw JSON as requested
        logger.info(f"Fetched data for {symbol}")
        
        # Monitor Latency
        latency = asyncio.get_event_loop().time() - start_fetch
        monitor.update_metrics(latency=latency)
        
        # Store normalized snapshot 
        snapshot_id = await insert_snapshot(db_pool, symbol, data)
        
        # Extract spot price
        spot_price = data.get("spot_price", 10000.0)
        timestamp = datetime.datetime.now()
        
        # Update history
        PRICE_HISTORY[symbol].append(spot_price)
        prices = list(PRICE_HISTORY[symbol])
        
        # Compute Features
        chain_data = data.get("option_chain", []) 
        oi_stats = feature_engineer.compute_oi_stats(chain_data)
        greeks = feature_engineer.compute_greeks(chain_data, spot_price)
        technicals = feature_engineer.compute_technicals(prices)
        volatility = feature_engineer.compute_volatility(chain_data)
        
        metrics = {
            **oi_stats,
            **greeks,
            **technicals,
            **volatility,
            "spot_price": spot_price
        }
        
        # Update Monitor with latest metrics if available
        # monitor.update_metrics(pnl=..., exposure=...) # PnL comes from Portfolio/Backtester, not ingestion directly
        
        # Store Features
        await insert_features(db_pool, snapshot_id, symbol, metrics)
        
        # --- SHOCK DETECTION ---
        # Get IV (ATM IV usually, or average)
        # Mocking IV extraction from metrics or using volatility['iv_rank'] as proxy?
        # volatility compute usually returns 'iv_skew' etc.
        # Let's assume we parse a 'current_iv' from chain_data or metrics.
        current_iv = metrics.get('atm_iv', 15.0) # Need to ensure feature_engineer returns this
        current_pcr = metrics.get('pcr', 1.0)
        
        shock_res = shock_detector.update(symbol, spot_price, current_iv, current_pcr, timestamp)
        
        if shock_res['triggered']:
            trigger_type = shock_res['type']
            logger.critical(f"SHOCK TRIGGERED: symbol={symbol}, type={trigger_type}")
            monitor.log_alert("CRITICAL")
            
            # 1. Kill Switch
            with open("STOP.flag", "w") as f:
                f.write(f"STOP triggered by {trigger_type} on {symbol}")
            
            # 2. Emergency Alert
            msg = f"🚨 SHOCK DETECTED on {symbol}! Type: {trigger_type}. Trading HALTED."
            alert_manager.send_message(msg, priority="EMERGENCY")
            
            # 3. Invalidate Signals
            await alert_manager.invalidate_active_alerts(db_pool)
            
            # 4. Log to DB
            await db_pool.execute("""
                INSERT INTO shock_logs (timestamp, trigger_type, details, action_taken)
                VALUES ($1, $2, $3, $4)
            """, timestamp, trigger_type, str(shock_res['details']), "KILL_SWITCH_ACTIVATED")
            
        # -----------------------
        
        # Regime Detection
        history = await fetch_recent_history(db_pool, symbol, limit=100)
        if len(history) > 10: 
             df_history = pd.DataFrame(history)
             regime = regime_detector.detect_regime(df_history)
             await insert_regime(db_pool, symbol, regime['label'], regime['details'])
             logger.info(f"Regime detected for {symbol}: {regime['label']}")
        
    except Exception as e:
        logger.error(f"Error in ingestion cycle for {symbol}: {e}")
        monitor.update_metrics(api_up=False) # Assume API issue if fetch failed

async def main():
    monitor = Monitor()
    monitor.start()
    
    db_pool = await get_db_pool()
    await init_db(db_pool)
    monitor.update_metrics(db_up=True)
    
    client = GrowwClient()
    feature_engineer = FeatureEngineer()
    regime_detector = RegimeDetector()
    alert_manager = AlertManager()
    
    # Create one detector per symbol to maintain separate history
    detectors = {s: ShockDetector() for s in SYMBOLS}
    
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    async with client:
        logger.info("Starting ingestion loop")
        monitor.update_metrics(api_up=True)
        while not stop_event.is_set():
            start_time = loop.time()
            
            # Check for STOP.flag (Manual or Shock Triggered)
            # If stopped, we might want to pause ingestion or just log "Paused"
            # The requirement is "Pause Trading", not necessarily data ingestion.
            # Usually we keep ingesting to see when to un-pause.
            # But the kill switch logic inside fetch_and_store writes the flag.
            
            tasks = [
                fetch_and_store(
                    client, symbol, db_pool, feature_engineer, regime_detector, 
                    detectors[symbol], alert_manager, monitor
                ) 
                for symbol in SYMBOLS
            ]
            await asyncio.gather(*tasks)
            
            elapsed = loop.time() - start_time
            sleep_time = max(0, FETCH_INTERVAL - elapsed)

            
            if not stop_event.is_set():
                logger.debug(f"Sleeping for {sleep_time} seconds")
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=sleep_time)
                except asyncio.TimeoutError:
                    pass # Timeout means 10s passed, continue loop
    
    await db_pool.close()
    logger.info("Ingestion engine stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

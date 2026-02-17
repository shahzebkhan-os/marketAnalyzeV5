import os
import requests
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class AlertManager:
    def __init__(self, telegram_bot_token: Optional[str] = None, telegram_chat_id: Optional[str] = None):
        self.bot_token = telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.last_alert_times: Dict[str, float] = {}
        
        # Risk Limits
        self.max_daily_loss = -5000.0 # Example: Stop if loss > 5000
        self.max_exposure = 50000.0   # Example: Max deployed capital
        
    def send_message(self, text: str, priority: str = "INFO"):
        """
        Sends a Telegram message.
        """
        if not self.bot_token or not self.chat_id:
            logger.warning(f"Telegram credentials not set. Message not sent: {text}")
            return

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": f"[{priority}] {text}",
                "parse_mode": "Markdown"
            }
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code != 200:
                logger.error(f"Failed to send Telegram message: status={response.status_code}, response={response.text}")
            else:
                logger.info(f"Telegram message sent: {text}")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")

    def check_risk(self, pnl: float, exposure: float) -> bool:
        """
        Checks risk limits. Returns True if risk breached (Kill Switch Trigger).
        """
        risk_breached = False
        
        # Check Daily Loss
        if pnl < self.max_daily_loss:
            msg = f"CRITICAL: Daily Loss Limit Breached! PnL: {pnl}"
            self.throttle_alert("risk_loss", msg, duration=300)
            risk_breached = True
            
        # Check Exposure
        if exposure > self.max_exposure:
            msg = f"WARNING: Max Exposure Limit Breached! Exposure: {exposure}"
            self.throttle_alert("risk_exposure", msg, duration=300)
            # Exposure breach might not kill switch immediately, but we flag it
            
        return risk_breached

    def monitor_drift(self, feature_name: str, current_value: float, historical_mean: float, threshold_std: float):
        """
        Simple Z-score check for feature drift.
        """
        # Checks if abs(current - mean) > 3 * std? 
        # Simplified: user passes 'threshold_std' as the value limit drift
        # e.g. if current > historical + threshold
        
        diff = abs(current_value - historical_mean)
        if diff > threshold_std:
            msg = f"DRIFT: {feature_name} divergence. Curr: {current_value:.2f}, Mean: {historical_mean:.2f}"
            self.throttle_alert(f"drift_{feature_name}", msg, duration=3600)

    def throttle_alert(self, key: str, message: str, duration: int = 60):
        """
        Sends alert only if not sent in last `duration` seconds.
        """
        now = time.time()
        last_time = self.last_alert_times.get(key, 0)
        
        if now - last_time > duration:
            self.send_message(message, priority="ALERT")
            self.last_alert_times[key] = now
            return True
        return False

    async def invalidate_active_alerts(self, pool):
        """
        Invalidates all currently OPEN alerts in the DB.
        Used during Shock Events to prevent stale signal execution.
        """
        # Assuming there is an 'alerts' table or 'signals' table.
        # If not, we log this action.
        # In this prototype, we don't have a 'signals' table explicitly defined in 'database.py' 
        # (we have option_chain, features, regimes, logs).
        # We will assume a 'trade_signals' table might exist or we just log.
        
        logger.warning("Invalidating all active alerts due to SHOCK event.")
        # Mock DB update
        # await pool.execute("UPDATE trade_signals SET status = 'INVALID' WHERE status = 'OPEN'")
        self.send_message("All active signals INVALIDATED due to market shock.", priority="EMERGENCY")


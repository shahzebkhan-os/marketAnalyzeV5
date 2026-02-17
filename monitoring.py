import time
import threading
from prometheus_client import start_http_server, Gauge, Histogram, Counter
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import logging

logger = logging.getLogger(__name__)

# --- Prometheus Metrics ---
SYSTEM_LATENCY = Histogram('system_ingestion_latency_seconds', 'Time spent in ingestion cycle')
MODEL_AUC = Gauge('model_auc_score', 'Current Model AUC Score', ['model_version'])
PORTFOLIO_PNL = Gauge('portfolio_pnl_total', 'Current Portfolio PnL')
RISK_EXPOSURE = Gauge('risk_exposure_total', 'Current Gross Exposure')
ALERT_COUNT = Counter('alert_sent_total', 'Total Alerts Sent', ['priority'])
DB_CONNECTION_STATUS = Gauge('db_connection_status', 'Database Connection Status (1=Up, 0=Down)')
API_CONNECTION_STATUS = Gauge('api_connection_status', 'Groww API Status (1=Up, 0=Down)')

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "status": "ok", 
                "timestamp": time.time()
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

class Monitor:
    def __init__(self, prometheus_port=8000, health_port=8080):
        self.prometheus_port = prometheus_port
        self.health_port = health_port
        self._health_server = None
    
    def start(self):
        """Starts Prometheus exporter and Health Check server in separate threads."""
        try:
            # Start Prometheus
            start_http_server(self.prometheus_port)
            logger.info(f"Prometheus metrics exposed on port {self.prometheus_port}")
            
            # Start Health Check
            self._start_health_server()
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")

    def _start_health_server(self):
        try:
            server = HTTPServer(('0.0.0.0', self.health_port), HealthCheckHandler)
            self._health_server = server
            thread = threading.Thread(target=server.serve_forever)
            thread.daemon = True
            thread.start()
            logger.info(f"Health check running on port {self.health_port}")
        except Exception as e:
            logger.error(f"Failed to start health check: {e}")

    def update_metrics(self, latency=None, auc=None, pnl=None, exposure=None, db_up=True, api_up=True):
        if latency is not None:
            SYSTEM_LATENCY.observe(latency)
        if auc is not None:
            MODEL_AUC.labels(model_version="latest").set(auc)
        if pnl is not None:
            PORTFOLIO_PNL.set(pnl)
        if exposure is not None:
            RISK_EXPOSURE.set(exposure)
        
        DB_CONNECTION_STATUS.set(1 if db_up else 0)
        API_CONNECTION_STATUS.set(1 if api_up else 0)

    def log_alert(self, priority):
        ALERT_COUNT.labels(priority=priority).inc()

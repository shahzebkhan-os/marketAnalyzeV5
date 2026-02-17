import asyncpg
import json
from settings import settings
from logger import get_logger

logger = get_logger(__name__)

async def get_db_pool():
    try:
        pool = await asyncpg.create_pool(dsn=settings.DATABASE_URL)
        logger.info("Database connection pool created")
        return pool
    except Exception as e:
        logger.error(f"Failed to create database connection pool: {e}")
        raise

async def init_db(pool):
    async with pool.acquire() as connection:
        # Create table if it doesn't exist
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS option_chain_snapshots (
                id BIGSERIAL PRIMARY KEY,
                symbol VARCHAR(50) NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                data JSONB NOT NULL
            );
        """)
        
        # Create index on symbol and timestamp
        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
            ON option_chain_snapshots (symbol, timestamp DESC);
        """)
        
        # Create table for derived features
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS derived_features (
                id BIGSERIAL PRIMARY KEY,
                snapshot_id BIGINT REFERENCES option_chain_snapshots(id),
                symbol VARCHAR(50) NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                metrics JSONB NOT NULL
            );
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_symbol_timestamp
            ON derived_features (symbol, timestamp DESC);
        """)
        
        # Create table for market regimes
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS market_regimes (
                id BIGSERIAL PRIMARY KEY,
                symbol VARCHAR(50) NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                regime_label VARCHAR(50) NOT NULL,
                details JSONB NOT NULL
            );
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_regimes_symbol_timestamp
            ON market_regimes (symbol, timestamp DESC);
        """)

        # Create table for drift logs
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS feature_drift_logs (
                id BIGSERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                feature_name VARCHAR(50) NOT NULL,
                metric_type VARCHAR(20) NOT NULL,
                value FLOAT NOT NULL,
                threshold FLOAT NOT NULL,
                status VARCHAR(20) NOT NULL
            );
        """)
        
        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_drift_timestamp
            ON feature_drift_logs (timestamp DESC);
        """)
        
        # Create table for model registry
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS model_registry (
                version_id SERIAL PRIMARY KEY,
                train_timestamp TIMESTAMPTZ DEFAULT NOW(),
                status VARCHAR(20) NOT NULL, -- ACTIVE, SHADOW, ARCHIVED
                metrics JSONB NOT NULL,
                filepath VARCHAR(255) NOT NULL,
                model_type VARCHAR(50) NOT NULL
            );
        """)

        # Create table for shadow evaluation logs
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS shadow_eval_logs (
                id BIGSERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                model_version_id INT REFERENCES model_registry(version_id),
                shadow_pnl FLOAT NOT NULL,
                production_pnl FLOAT NOT NULL
            );
        """)

        # Create table for risk simulation logs
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS risk_simulation_logs (
                id BIGSERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                scenarios JSONB NOT NULL, -- {var95, var99, worst_case}
                stress_results JSONB NOT NULL -- {gap_up, gap_down, iv_spike}
            );
        """)

        # Create table for explainability logs
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS explainability_logs (
                id BIGSERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                model_name VARCHAR(50) NOT NULL,
                instance_id VARCHAR(50), -- Optional ID to link to specific prediction
                shap_values JSONB, -- {feature: value}
                top_features JSONB, -- List of top contributing features
                attention_weights JSONB -- For sequence models
            );
        """)

        # Create table for shock logs
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS shock_logs (
                id BIGSERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                trigger_type VARCHAR(50) NOT NULL, -- PRICE_MOVE, IV_SPIKE, PCR_ANOMALY
                details JSONB NOT NULL,
                action_taken VARCHAR(100) -- KILL_SWITCH, RISK_REDUCED, etc.
            );
        """)

        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_symbol_timestamp
            ON derived_features (symbol, timestamp DESC);
        """)
        
        logger.info("Database schema initialized")

async def insert_snapshot(pool, symbol: str, data: dict):
    try:
        async with pool.acquire() as connection:
            # Insert and return ID
            row = await connection.fetchrow("""
                INSERT INTO option_chain_snapshots (symbol, data)
                VALUES ($1, $2::jsonb)
                RETURNING id
            """, symbol, json.dumps(data))
            logger.info(f"Snapshot inserted successfully for {symbol}, id={row['id']}")
            return row['id']
    except Exception as e:
        logger.error(f"Failed to insert snapshot for {symbol}: {e}")
        raise

async def insert_features(pool, snapshot_id: int, symbol: str, metrics: dict):
    try:
        async with pool.acquire() as connection:
            await connection.execute("""
                VALUES ($1, $2, $3::jsonb)
            """, snapshot_id, symbol, json.dumps(metrics))
            logger.info(f"Features inserted successfully for {symbol}")
    except Exception as e:
        logger.error(f"Failed to insert features for {symbol}: {e}")
        raise

async def insert_regime(pool, symbol: str, label: str, details: dict):
    try:
        async with pool.acquire() as connection:
            await connection.execute("""
                INSERT INTO market_regimes (symbol, regime_label, details)
            """, symbol, label, json.dumps(details))
            logger.info(f"Regime inserted successfully for {symbol}: {label}")
    except Exception as e:
        logger.error(f"Failed to insert regime for {symbol}: {e}")
        pass

async def insert_drift_log(pool, feature_name: str, metric_type: str, value: float, threshold: float, status: str):
    try:
        async with pool.acquire() as connection:
            await connection.execute("""
                INSERT INTO feature_drift_logs (feature_name, metric_type, value, threshold, status)
                VALUES ($1, $2, $3, $4, $5)
            """, feature_name, metric_type, value, threshold, status)
            # logger.info(f"Drift log inserted for {feature_name}: {status}")
    except Exception as e:
        logger.error(f"Failed to insert drift log for {feature_name}: {e}")
        pass

async def register_model(pool, status: str, metrics: dict, filepath: str, model_type: str):
    try:
        async with pool.acquire() as connection:
            row = await connection.fetchrow("""
                INSERT INTO model_registry (status, metrics, filepath, model_type)
                VALUES ($1, $2::jsonb, $3, $4)
            """, status, json.dumps(metrics), filepath, model_type)
            logger.info(f"Model registered: version={row['version_id']}, status={status}")
            return row['version_id']
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        pass

async def log_shadow_eval(pool, version_id: int, shadow_pnl: float, prod_pnl: float):
    try:
        async with pool.acquire() as connection:
            await connection.execute("""
                INSERT INTO shadow_eval_logs (model_version_id, shadow_pnl, production_pnl)
                VALUES ($1, $2, $3)
            """, version_id, shadow_pnl, prod_pnl)
    except Exception as e:
        logger.error(f"Failed to log shadow eval: {e}")
        pass

async def update_model_status(pool, version_id: int, new_status: str):
    try:
        async with pool.acquire() as connection:
            await connection.execute("""
                UPDATE model_registry
                SET status = $1
                WHERE version_id = $2
            """, new_status, version_id)
            logger.info(f"Model status updated for version {version_id}: {new_status}")
    except Exception as e:
        logger.error(f"Failed to update model status: {e}")
        pass

async def fetch_recent_history(pool, symbol: str, limit: int = 100) -> list:
    """
    Fetches recent price and feature data for regime detection.
    Returns list of dicts with keys: timestamp, close, atr, iv.
    """
    try:
        query = """
            SELECT 
                s.timestamp,
                (s.data->>'spot_price')::float as close,
                (f.metrics->>'atr')::float as atr,
                (f.metrics->>'iv')::float as iv
            FROM option_chain_snapshots s
            JOIN derived_features f ON s.id = f.snapshot_id
            WHERE s.symbol = $1
            ORDER BY s.timestamp DESC
            LIMIT $2
        """
        async with pool.acquire() as connection:
            rows = await connection.fetch(query, symbol, limit)
            # Return reversed to be in chronological order
            return [dict(r) for r in reversed(rows)]
    except Exception as e:
        logger.error(f"Failed to fetch history for {symbol}: {e}")
        return []
    return []

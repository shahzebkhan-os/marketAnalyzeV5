# Holographic Glenn - Trading Dashboard

A real-time intraday options intelligence engine and dashboard.

## Overview
This project consists of two main components:
1.  **Ingestion Engine**: Fetches market data (currently via Groww API), computes features/greeks, detects regimes, and stores data in PostgreSQL.
2.  **Dashboard**: An interactive Streamlit frontend for monitoring market conditions, alerts, and model performance.

## Prerequisites
-   Python 3.11+
-   Docker Desktop (for the full backend system)

## Quick Start (Dashboard Only)
You can run the dashboard in standalone mode (with mock data) without starting the entire infrastructure.

1.  **Run Dashboard**
    Use the python executable from your virtual environment:
    ```bash
    ./.venv/bin/python -m streamlit run dashboard.py
    ```
    Access the dashboard at `http://localhost:8501`.

    *Note: If you have already activated your virtual environment (`source .venv/bin/activate`), you can simply run `streamlit run dashboard.py`.*

2.  **Install Dependencies (if not already installed)**
    ```bash
    ./.venv/bin/pip install -r requirements.txt
    ```

## Full System Setup (Ingestion + DB + Monitoring)
To run the full data pipeline, including the database and ingestion service:

1.  **Environment Setup**
    Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
    Edit `.env` and fill in your `GROWW_API_KEY`, `GROWW_API_SECRET`, etc.

2.  **Start Services**
    Use Docker Compose.
    
    If you are using a recent version of Docker Desktop:
    ```bash
    docker compose up --build -d
    ```
    
    If that command is not found, try the legacy command:
    ```bash
    docker-compose up --build -d
    ```

3.  **Service Endpoints**
    -   **Ingestion Service Health**: `http://localhost:8080/health`
    -   **Prometheus**: `http://localhost:9090`
    -   **Grafana**: `http://localhost:3000` (Default login: `admin` / `admin`)
    -   **PostgreSQL**: `localhost:5432`

## Project Structure
-   `dashboard.py`: Main Streamlit application.
-   `ingestion.py`: Async data ingestion worker.
-   `features.py`: Feature engineering logic (Greeks, Technicals).
-   `risk_engine.py`: Risk management and position sizing.
-   `docker-compose.yml`: Container orchestration configuration.

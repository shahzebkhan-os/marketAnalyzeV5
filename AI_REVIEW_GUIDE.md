# AI Project Review Guide: Intraday Options Intelligence Engine

This document is structured for an external AI Agent to quickly audit, review, and understand the core logic of this project.

## 1. Project Objectives
- **System:** Real-Time F&O Options Scanner and Intelligence Engine.
- **Goal:** Identify high-confidence intraday trading setups using OI concentration and Greek exposure.
- **Key Metric:** Model Confidence based on institutional positioning (Max OI).

## 2. Core Components to Audit

### Data Layer (`client.py`)
- **Review Item:** `GrowwClient` implementation.
- **Logic:** Uses a hybrid approach (SDK + Web Fallback) to fetch option chains and market scans.
- **Audit Focus:** Rate limiting handling, token lifecycle management, and payload structure for `fetch_option_chain`.

### Intelligence Logic (`dashboard.py` -> `analyze_candidate`)
    - **Bearish:** `if type == 'CE'` (Call Resistance).
- **Audit Focus:** Sentiment scoring logic, confidence weighting (OI Dominance vs. Prox Score), and real-time refresh orchestration.

### Antigravity Protocol (`features.py` -> `calculate_antigravity`)
- **Review Item:** `calculate_antigravity(chain_df, spot_price)`
- **Logic:** Weighted score (Trap 40%, Panic 40%, Velocity 20%) detecting short covering.
- **Audit Focus:** `oi_change` extraction from `client.py` and the normalization of the "Panic" ratio.

### Feature Engineering (`features.py`)
- **Review Item:** `FeatureEngineer` class.
- **Calculations:** PCR, GEX, DEX, Max Pain, and Black-Scholes Greeks.
- **Audit Focus:** Correctness of Black-Scholes implementation, vectorized GEX calculations, and technical indicator robustness.

### Dashboard Orchestration (`dashboard.py`)
- **Review Item:** `tab2` (Market Scanner) logic.
- **Logic:** `Batch Analyze ALL` performs a sequential scan across 200+ underlyings.
- **Audit Focus:** Async loop management (`asyncio.new_event_loop`), Streamlit's `st.empty()` for real-time table flushing, and session state persistence.

## 3. Critical Flow: Full Market Scan
1. **Trigger:** `run_market_scan()` fetches all F&O symbols.
2. **Analysis:** `Batch Analyze ALL` iterates through symbols.
3. **Execution:** Each symbol calls `analyze_candidate`, which fetches a full 100-strike option chain.
4. **Presentation:** Results are sorted by `spot` price and displayed with `Trend` and `Confidence` metrics.

## 4. Potential Improvements for AI to Propose
- **Performance:** Parallelize `analyze_candidate` with strict rate-limit aware semaphores using `asyncio.gather`.
- **Advanced Features:** Implement Gamma Flip level detection and Volatility Surface visualization.
- **Risk:** Integration with `RiskEngine` for automated position sizing based on scan results.

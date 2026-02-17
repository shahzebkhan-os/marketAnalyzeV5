# Real-Time F&O Market Scanner: Analysis Framework

This document provides a detailed technical explanation of the analysis process and framework used in the Real-Time F&O Market Scanner.

## 1. Core Framework Overview

The scanner is built on a modular Python architecture designed for high-performance market data processing. It combines real-time data ingestion, advanced feature engineering, and a rules-based intelligence engine to identify high-probability setups in the F&O segment.

### Technology Stack
- **Backend:** Python 3.9+
- **Data Ingestion:** Groww API (Custom Client)
- **Data Processing:** Pandas, NumPy, SciPy
- **UI & Visualization:** Streamlit, Plotly
- **Concurrency:** Asyncio for non-blocking I/O

---

## 2. The Analysis Process

The analysis follows a sequential multi-stage pipeline:

### Stage 1: Data Ingestion
- **Market Scan:** The system first fetches all liquid F&O underlyings from the exchange.
- **Spot Prices:** It fetches the Last Traded Price (LTP) for all symbols in batch to optimize network calls.
- **Option Chains:** For each interesting candidate, it fetches the full depth of the option chain (OI, IV, Bid/Ask, Greeks).

### Stage 2: Feature Engineering
The `FeatureEngineer` class computes critical metrics from the raw data:
- **OI Statistics:**
    - **Put-Call Ratio (PCR):** Total Put OI divided by Total Call OI. Values > 1 suggest bullish sentiment (support), while < 1 suggest bearish sentiment (resistance).
    - **OI Imbalance:** The net difference between Call and Put OI.
    - **Max Pain:** The strike price where option writers (sellers) lose the least amount of money, often acting as a magnet for the spot price.
- **Option Greeks (Black-Scholes):**
    - **Delta:** Sensitivity of option price to changes in the underlying price.
    - **Gamma:** Rate of change of Delta; critical for understanding dealer hedging pressure.
    - **GEX (Gamma Exposure):** Monitors how much market makers need to hedge as prices move.
- **Technical Indicators:** RSI, MACD, EMA (20/50), and Bollinger Bands are computed for trend verification.

### Stage 3: Intelligence & Sentiment Scoring
The `analyze_candidate` engine evaluates each stock using the following rules:
1. **Selection:** Identify the strike price with the **Highest Open Interest (OI)** in the entire chain.
2. **Sentiment Identification:**
    - If the Max OI is in a **Put (PE)**, the signal is **BULLISH** (Put Support).
    - If the Max OI is in a **Call (CE)**, the signal is **BEARISH** (Call Resistance).
3. **Confidence Calculation:**
    - **OI Score (60%):** How "dominant" the top strike's OI is relative to the rest of the chain.
    - **Proximity Score (40%):** How close the candidate strike is to the current Spot Price (ATM proximity).
4. **Final Recommendation:**
    - **Strong Bull/Bear:** Confidence > 80%
    - **Bullish/Bearish:** Confidence > 60%
    - **Watch:** Confidence < 60%

---

## 3. Real-Time Dashboard Integration

The Streamlit dashboard (`dashboard.py`) manages the orchestration:
- **Async Batching:** Analyzes multiple symbols in a non-blocking loop.
- **Dynamic UI Updates:** Uses Streamlit placeholders to refresh the market table as each result is computed, providing immediate feedback during long scans.
- **Refresh Cycle:** Automatically reruns the dashboard based on a user-defined interval to keep data fresh.

---

## 4. Safety and Risk Controls
The framework includes a **Kill Switch** and **Shock Detector** that monitors for extreme IV spikes or massive drift in feature values, halting execution if risk parameters are breached.

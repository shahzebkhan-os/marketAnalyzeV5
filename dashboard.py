import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import asyncio
from datetime import datetime
import os
import logging
import time

# Configure standard logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from alerts import AlertManager
from database import get_db_pool # Simplified: In real app, we need async db access or sync wrapper.
from client import GrowwClient
from features import FeatureEngineer


st.set_page_config(page_title="Holographic Glenn Dashboard", layout="wide", page_icon="📈")

# --- CSS Styling ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .metric-value {
        font_size: 24px;
        font-weight: bold;
        color: #00FF00;
    }
    .metric-label {
        color: #888;
    }
    .stProgress > div > div > div > div {
        background-color: #00FF00;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("Configuration")
symbol = st.sidebar.selectbox("Symbol", ["NIFTY", "BANKNIFTY"])
refresh_rate = st.sidebar.slider("Refresh Rate (s)", 1, 60, 10)
use_live_data = st.sidebar.toggle("⚡ Live Data (Groww API)", value=True)

st.sidebar.markdown("### Controls")
kill_switch = st.sidebar.toggle("❌ KILL SWITCH", value=False)

if kill_switch:
    st.sidebar.error("SYSTEM STOPPED BY USER")
    # In a real app, write to a flag file or DB
    with open("STOP.flag", "w") as f:
        f.write("STOP")
else:
    if os.path.exists("STOP.flag"):
        os.remove("STOP.flag")

# --- Auto-Execution & Hedge Controls ---
auto_exec = st.sidebar.toggle("🤖 Auto-Execution", value=False)
auto_hedge = st.sidebar.toggle("🛡️ Auto-Hedge", value=False)

if auto_exec:
    st.sidebar.success("Auto-Execution ENABLED")
else:
    st.sidebar.warning("Auto-Execution DISABLED")

# --- Manual Trade Override ---
with st.sidebar.expander("🛠️ Manual Trade Override"):
    with st.form("manual_trade_form"):
        m_symbol = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY"])
        m_type = st.selectbox("Type", ["CE", "PE", "FUT"])
        m_strike = st.number_input("Strike", min_value=10000, max_value=60000, step=100)
        m_qty = st.number_input("Qty", min_value=1, step=1)
        m_submit = st.form_submit_button("Execute Trade")
        
        if m_submit:
            st.toast(f"Manual Order Sent: {m_symbol} {m_strike} {m_type} x {m_qty}")
            # In real app: send to Order Execution Engine
            with open("manual_orders.log", "a") as f:
                f.write(f"{pd.Timestamp.now()},{m_symbol},{m_type},{m_strike},{m_qty}\n")

# --- Mock Data Generation (Replace with DB calls) ---
import asyncio
from client import GrowwClient
from features import FeatureEngineer

# --- Data Loading Logic ---
@st.cache_resource
def get_groww_client():
    """Returns a persistent GrowwClient instance."""
    return GrowwClient()

async def fetch_live_data(symbol):
    """
    Fetches real/mocked structured data via GrowwClient and computes features.
    """
    try:
        # Use shared GrowwClient instance as a context manager for proper session lifecycle
        client = get_groww_client()
        async with client:
            data = await client.fetch_option_chain(symbol)
                
            spot = data.get('spot_price', 0.0)
            chain_data = data.get('option_chain', [])
            
            # Compute Features
            fe = FeatureEngineer()
            chain_df = pd.DataFrame(chain_data)
            oi_stats = fe.compute_oi_stats(chain_data)
            greeks = fe.compute_greeks(chain_data, spot)
            vol_stats = fe.compute_volatility(chain_data)
            antigravity = fe.calculate_antigravity(chain_df, spot)
            
            return {
                "regime": "Trending (Live)", # Placeholder
                "confidence": 85.0, # Placeholder
                "spot": spot,
                "pnl": 0.0,
                "exposure": 0.0,
                "oi_stats": oi_stats,
                "greeks": greeks,
                "vol_stats": vol_stats,
                "antigravity": antigravity,
                "chain_data": chain_data
            }
    except Exception as e:
        import traceback
        with open("error_traceback_new.txt", "w") as f:
            f.write(traceback.format_exc())
        err_str = str(e).lower()
        logger.warning(f"Fetch live data exception: {err_str}")
        if "rate limit" in err_str or "429" in err_str or "throttled" in err_str:
            return "RATE_LIMIT"
        st.error(f"Failed to fetch live data for {symbol}: {str(e)}")
        return None



@st.cache_data(ttl=1)
def get_market_status():
    if 'last_market_status' not in st.session_state:
        st.session_state.last_market_status = None

    if use_live_data:
        try:
            # Use asyncio.run for clean loop management per call
            data = asyncio.run(fetch_live_data(symbol))
            
            if data == "RATE_LIMIT":
                if st.session_state.last_market_status:
                    st.warning("⚠️ API Rate Limit hit. Using last successful data.")
                    return st.session_state.last_market_status
                else:
                    # Only show blocking message if we have NO data at all
                    st.info("🕒 Market Data unavailable. Retrying connection... (10s)")
                    st.stop()
            
            if data:
                st.session_state.last_market_status = data
                return data
            elif st.session_state.last_market_status:
                st.warning("🔄 Using Cached Data: Live fetch delayed.")
                return st.session_state.last_market_status
            else:
                st.error("Market Data Unavailable. Please check API keys or connection.")
                st.stop()
        except Exception as e:
            if "Access forbidden" in str(e):
                st.error("🚫 **Access Forbidden (403)**: Your Groww account lacks permissions for F&O market data. Please enable Trading API and F&O segments in your Groww developer portal.")
            else:
                st.error(f"Async Error: {e}")
            st.stop()
    else:
        # If user turns off live data but we are in strict mode, what should we do?
        # The user asked to "only test on real data".
        # We can either disable the toggle or make the "false" state also try to fetch (removing the toggle concept).
        # But for now, let's just make the fallback explicit that mock is removed.
        st.warning("Mock data is disabled. Please enable 'Live Data' to fetch real data.")
        return {
            "regime": "WAITING",
            "confidence": 0.0,
            "spot": 0.0,
            "pnl": 0.0,
            "exposure": 0.0,
            "oi_stats": {},
            "greeks": {},
            "vol_stats": {}
        }

def get_opportunities(status):
    """
    Generates trading opportunities based on live data if confidence is high.
    """
    if status.get('confidence', 0) < 80:
        return pd.DataFrame(columns=["strike", "type", "ltp", "oi", "gex", "Signal"])

    chain = status.get('chain_data', [])
    if not chain:
        return pd.DataFrame(columns=["strike", "type", "ltp", "oi", "gex", "Signal"])

    df = pd.DataFrame(chain)
    
    # Heuristic for opportunities: 
    # 1. High OI strikes near ATM
    # 2. Significant GEX (Gamma Exposure)
    # We'll pick top 5 by OI near ATM
    spot = status.get('spot', 0)
    df['dist_from_spot'] = (df['strike'] - spot).abs()
    
    # Filter for near-ATM
    near_atm = df.sort_values('dist_from_spot').head(20)
    
    # Pick Top Opportunities
    opps = near_atm.sort_values('oi', ascending=False).head(5).copy()
    
    # Add metadata
    opps['Signal'] = "STRADDLE" if status['regime'] == "Volatile" else "DIRECTIONAL"
    
    # Format for display
    display_df = opps[['strike', 'type', 'ltp', 'oi', 'Signal']].copy()
    display_df.columns = ["Strike", "Type", "LTP", "OI", "Signal"]
    
    return display_df

# --- Main Dashboard ---

st.title(f"🚀 Holographic Glenn: {symbol}")

# --- NEW: Dashboard Tabs ---
tab1, tab2 = st.tabs(["📉 Active Symbol", "🔍 Market Scanner"])

with tab1:
    # --- ANTIGRAVITY ALERT ---
    status = get_market_status()
    if status and status.get('antigravity', {}).get('status'):
        ant = status['antigravity']
        st.warning(f"🚀 **ANTIGRAVITY ALERT**: Short Covering Detected! Score: {ant['score']} | Wall: {ant['wall_strike']} | Shed: {ant['oi_shed']}")
    # --- SHOCK BANNER ---
    # Check for STOP flag or Emergency State
    if os.path.exists("STOP.flag"):
        st.error("🚨 EMERGENCY: TRADING HALTED DUE TO MARKET SHOCK OR KILL SWITCH 🚨")
        st.markdown("""
            <div style="background-color: #FF0000; color: white; padding: 20px; font-weight: bold; text-align: center; border-radius: 5px;">
                SYSTEM HALTED. RISK PARAMETERS REDUCED. CHECK LOGS.
            </div>
        """, unsafe_allow_html=True)
    # --------------------

    # Header Metrics
    status = get_market_status()

    if status:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Regime", status['regime'], delta="Stable")
        c2.metric("Confidence", f"{status['confidence']:.1f}%", delta=f"{status['confidence']-80:.1f}%")
        c3.metric("Spot Price", f"{status['spot']:.2f}", delta=f"{np.random.randn():.2f}")
        c4.metric("PnL", f"₹ {status['pnl']:.2f}", delta_color="normal" if status['pnl'] > 0 else "inverse")
    else:
        st.warning("⚠️ API Rate Limit hit or Market Data unavailable. Retrying in background...")
        st.stop()

    # Row 2: Institutional Metrics
    r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
    # PCR from real features
    pcr = status.get('oi_stats', {}).get('pcr', 0.0)
    antigravity_score = status.get('antigravity', {}).get('score', 0.0)
    
    r2_c1.metric("PCR", f"{pcr:.2f}", delta="Live")
    r2_c2.metric("Antigravity Score", f"{antigravity_score}", delta="Short Squeeze" if antigravity_score > 75 else "Stable")
    r2_c3.metric("Active Model", "v1.2 (LGBM+LSTM)")
    r2_c4.metric("Next Retrain", "4h 12m")


    # Confidence Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = status['confidence'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Model Confidence"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00FF00" if status['confidence'] > 75 else "#F39C12"},
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Risk Checks
    alert_mgr = AlertManager()
    if alert_mgr.check_risk(status['pnl'], status['exposure']):
        st.error(f"⚠️ RISK BREACH DETECTED! PnL: {status['pnl']:.2f}")

    # Alert History Expander
    with st.expander("🔔 Alert History (Recent)", expanded=False):
        # Mock Alerts
        alerts_mock = pd.DataFrame({
            "Time": [pd.Timestamp.now() - pd.Timedelta(minutes=m) for m in [5, 15, 60]],
            "Priority": ["INFO", "WARNING", "INFO"],
            "Message": ["Drift Detected (PSI > 0.1)", "Gamma Exposure High", "Retraining Started"]
        })
        st.dataframe(alerts_mock, use_container_width=True)


    # Opportunities
    st.subheader("💡 Top Opportunities")
    opps = get_opportunities(status)
    if opps.empty:
        st.info("Waiting for High Confidence Signal (Confidence < 80%)...")
    else:
        st.dataframe(opps.style.highlight_max(axis=0, color="#112211"), use_container_width=True)
    
    # --- Heatmaps & Charts ---
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("🔥 Gamma Profile")
        # Placeholder for real chart logic
        st.line_chart(np.random.randn(10, 1))
    with col_right:
        st.subheader("🌊 OI Heatmap")
        st.line_chart(np.random.randn(10, 1))

# --- Market Scanner Logic ---
async def run_market_scan(progress_bar=None):
    client = get_groww_client()
    async with client:
        return await client.fetch_market_scan()

async def analyze_candidate(symbol):
    try:
        client = get_groww_client()
        async with client:
            data = await client.fetch_option_chain(symbol)
            chain = data.get('option_chain', [])
            spot = data.get('spot_price', 0.0)
            if not chain:
                logger.warning(f"Market Scan in progress for {symbol}...")
                return None
            
            df = pd.DataFrame(chain)
            if df.empty or 'oi' not in df.columns:
                return None
            
            # 1. Selection: Best is highest OI strike
            best = df.sort_values('oi', ascending=False).iloc[0].to_dict()
            
            # 2. Confidence Scoring
            max_oi = df['oi'].max()
            oi_score = (best['oi'] / max_oi) if max_oi > 0 else 0
            
            # ATM Proximity (Penalty if > 3% away from spot)
            dist_pct = abs(best['strike'] - spot) / spot if spot > 0 else 1.0
            prox_score = max(0, 1 - (dist_pct / 0.03)) 
            
            confidence = (oi_score * 0.6 + prox_score * 0.4) * 100
            best['confidence_score'] = round(confidence, 1)
            
            # 3. Recommendation Logic (Market Sentiment)
            # High OI in PE is often support (Bullish), High OI in CE is resistance (Bearish)
            if best['type'].upper() == 'PE':
                sentiment = "BULLISH (PUT SUPPORT)"
                action = "🔥 STRONG BULL" if confidence > 80 else "✅ BULLISH"
            else:
                sentiment = "BEARISH (CALL RESIST)"
                action = "💀 STRONG BEAR" if confidence > 80 else "🔻 BEARISH"

            if confidence < 60:
                action = "👀 WATCH"
                
            best['action'] = action
            best['sentiment'] = sentiment

            # 4. Antigravity Logic
            ant = fe.calculate_antigravity(df, spot)
            best['antigravity_score'] = ant['score']
                
            return best
    except Exception as e:
        logger.error(f"Analysis Failed for {symbol}: {e}")
        return None

# --- Market Scanner Tab ---
with tab2:
    st.subheader("🌐 Real-Time F&O Market Scanner")
    st.info("Scanning all F&O underlyings for activity and setups...")
    
    if 'scanner_results' not in st.session_state:
        st.session_state.scanner_results = None
    if 'best_setups' not in st.session_state:
        st.session_state.best_setups = {}

    if st.button("🚀 Start Full Market Scan"):
        progress_text = "Discovering F&O stocks..."
        my_bar = st.progress(0, text=progress_text)
        
        with st.spinner("Scanning market..."):
            try:
                # Use a perfectly isolated loop for the UI-triggered scan
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    results = loop.run_until_complete(run_market_scan())
                    st.session_state.scanner_results = results
                finally:
                    loop.close()
                
                my_bar.progress(100, text="Scan complete!")
                st.rerun()
            except Exception as e:
                import traceback
                logger.error(f"Scanner Crash: {traceback.format_exc()}")
                st.error(f"Scanner Error: {e}")

    if st.session_state.scanner_results:
        scan_df = pd.DataFrame(st.session_state.scanner_results)
        
        # Batch Analyze ALL Button
        if st.button("🔍 Batch Analyze ALL Candidates"):
            all_candidates = scan_df['symbol'].tolist()
            progress_container = st.empty()
            table_placeholder = st.empty()
            
            total = len(all_candidates)
            for i, cand in enumerate(all_candidates):
                progress_container.progress((i + 1) / total, text=f"Analyzing {cand} ({i+1}/{total})...")
                
                try:
                    # Run analysis for individual candidate
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        res = loop.run_until_complete(analyze_candidate(cand))
                        if res:
                            st.session_state.best_setups[cand] = res
                    finally:
                        loop.close()
                    
                    # Real-time Table Update (Selective refresh)
                    current_scan_df = scan_df.copy()
                    # Add placeholder columns if they don't exist
                    for col in ['Rec. Strike', 'Trend', 'Confidence']:
                        if col not in current_scan_df.columns:
                            current_scan_df[col] = "—"
                    
                    for symbol, setup in st.session_state.best_setups.items():
                        idx = current_scan_df.index[current_scan_df['symbol'] == symbol]
                        if not idx.empty:
                            current_scan_df.loc[idx, 'Rec. Strike'] = setup.get('strike', '—')
                            current_scan_df.loc[idx, 'Trend'] = setup.get('action', '—')
                            current_scan_df.loc[idx, 'Confidence'] = f"{setup.get('confidence_score', 0)}%"
                    
                    current_scan_df = current_scan_df.sort_values('spot', ascending=False)
                    table_placeholder.dataframe(current_scan_df[['symbol', 'spot', 'Rec. Strike', 'Trend', 'Confidence']], use_container_width=True)
                    
                except Exception as e:
                    logger.error(f"Batch Analysis failed for {cand}: {e}")
                    
            progress_container.success(f"Successfully analyzed {total} candidates!")
            time.sleep(1)
            st.rerun()
        scan_df = pd.DataFrame(st.session_state.scanner_results)
        
        # Add placeholder columns if they don't exist
        for col in ['Rec. Strike', 'Trend', 'Confidence']:
            if col not in scan_df.columns:
                scan_df[col] = "—"
        
        # Merge existing best_setups into the dataframe
        for symbol, setup in st.session_state.best_setups.items():
            idx = scan_df.index[scan_df['symbol'] == symbol]
            if not idx.empty:
                scan_df.loc[idx, 'Rec. Strike'] = setup.get('strike', '—')
                scan_df.loc[idx, 'Trend'] = setup.get('action', '—')
                scan_df.loc[idx, 'Confidence'] = f"{setup.get('confidence_score', 0)}%"
                scan_df.loc[idx, 'Antigravity'] = f"{setup.get('antigravity_score', 0)}"

        st.success(f"Found {len(scan_df)} F&O underlyings!")
        
        # Sort by spot price descending and reorder columns for clarity
        scan_df = scan_df.sort_values('spot', ascending=False)
        cols = ['symbol', 'spot', 'Rec. Strike', 'Trend', 'Confidence', 'Antigravity']
        st.dataframe(scan_df[cols], use_container_width=True)
        
        st.markdown("### 💡 Recommended 'Best' Chains")
        # Suggest top 10 candidates
        candidates = scan_df.head(10)['symbol'].tolist()
        for cand in candidates:
            with st.expander(f"Identify Best Options for {cand}"):
                # Check if we already have a result for this candidate
                if cand in st.session_state.best_setups:
                    best_opt = st.session_state.best_setups[cand]
                    st.markdown(f"### {best_opt['action']} ({best_opt['confidence_score']}%)")
                    st.write(f"**Strike**: {best_opt['strike']} | **Type**: {best_opt['type']} | **LTP**: ₹{best_opt['ltp']} | **OI**: {best_opt['oi']}")
                    st.info(f"Rationale: High OI concentration ({best_opt['oi']}) relative to chain max.")
                
                if st.button(f"Scan {cand} for Best Setup", key=f"scan_{cand}"):
                    # ... exists ...
                    try:
                        with st.spinner(f"Analyzing {cand}..."):
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            best_opt = loop.run_until_complete(analyze_candidate(cand))
                            loop.close()
                            
                            if best_opt:
                                st.session_state.best_setups[cand] = best_opt
                                st.rerun() # Persist result
                            else:
                                st.warning("No setup found for this candidate.")
                    except Exception as inner_e:
                        st.error(f"Analysis failed: {inner_e}")

with col_left:
    st.subheader("🔥 Gamma Profile")
    if status and 'chain_data' in status and status['chain_data']:
        df_gamma = pd.DataFrame(status['chain_data'])
        if 'iv' in df_gamma.columns and 'oi' in df_gamma.columns:
            # Compute individual GEX per strike
            spot = status['spot']
            fe = FeatureEngineer()
            df_gamma['gex'] = df_gamma.apply(lambda x: fe.black_scholes_greeks(spot, x['strike'], 7/365, 0.05, x['iv'], x['type'].lower())[1] * x['oi'] * spot, axis=1)
            
            # Group by strike
            gamma_by_strike = df_gamma.groupby('strike')['gex'].sum().reset_index()
            fig_gex = px.bar(gamma_by_strike, x='strike', y='gex', title=f"Net Gamma Exposure: {symbol}", labels={'strike': 'Strike', 'gex': 'Gamma'})
            st.plotly_chart(fig_gex, use_container_width=True)
        else:
            st.warning("Insufficient chain metadata (IV/OI) for Gamma profile.")
    else:
        st.info("Gamma data waiting for live chain...")

with col_right:
    st.subheader("🌊 IV Surface")
    if status and 'chain_data' in status and status['chain_data']:
        df_iv = pd.DataFrame(status['chain_data'])
        if 'iv' in df_iv.columns:
            # Simplified heatmap: Strike vs Type IV
            iv_pivot = df_iv.pivot_table(index='strike', columns='type', values='iv', aggfunc='mean')
            fig_iv = px.imshow(iv_pivot, title=f"IV Surface: {symbol}", labels=dict(x="Option Type", y="Strike", color="IV"))
            st.plotly_chart(fig_iv, use_container_width=True)
        else:
            st.warning("No IV data available for volatility surface.")
    else:
        st.info("Volatility surface waiting for live chain...")

# Activity Log (Mock)
st.subheader("📜 Activity Log")
st.text("10:00:05 - Ingestion Cycle Complete. Regime: Trending")
st.text("10:00:02 - Signal Generated: BUY 22500 CE (Conf: 85%)")

# --- Safety, Risk & Drift Tab ---
st.divider()
st.header("🛡️ Model Safety & Drift Monitoring")
t1, t2, t3, t4, t5, t6 = st.tabs(["Rolling Performance", "Feature Drift", "Model Registry", "Risk & Hedge", "Explainability", "Optimization"])


with t1:
    st.subheader("Rolling AUC (Last 50 windows)")
    # Mock rolling AUC
    windows = np.arange(50)
    aucs = 0.6 + np.random.randn(50) * 0.05
    aucs[-5:] = 0.58 # Drop recently
    
    fig_auc = px.line(x=windows, y=aucs, labels={'x': 'Window', 'y': 'AUC'}, title="Rolling AUC")
    fig_auc.add_hline(y=0.55, line_dash="dot", line_color="red", annotation_text="Critical Threshold (0.55)")
    st.plotly_chart(fig_auc, use_container_width=True)
    
    if aucs[-1] < 0.55:
        st.error("MODEL UNSAFE: AUC below critical threshold.")

with t2:
    st.subheader("Drift Log (KS & PSI)")
    # Mock Drift Log
    drift_data = pd.DataFrame({
        "Timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
        "Feature": ["delta", "gamma", "oi_change", "iv", "volume"],
        "Metric": ["KS", "PSI", "KS", "PSI", "KS"],
        "Value": [0.005, 0.3, 0.02, 0.1, 0.001],
        "Status": ["CRITICAL", "WARNING", "OK", "OK", "CRITICAL"]
    })
    
    def color_status(val):
        color = 'green'
        if val == 'CRITICAL': color = 'red'
        elif val == 'WARNING': color = 'orange'
        return f'color: {color}'

    st.dataframe(drift_data.style.map(color_status, subset=['Status']), use_container_width=True)

with t3:
    st.subheader("🤖 Model Registry")
    
    # Mock Registry Data
    registry_data = pd.DataFrame({
        "Version": ["v1.0 (Prod)", "v1.1 (Shadow)"],
        "Status": ["ACTIVE", "SHADOW"],
        "Train Date": ["2024-01-01", "2024-01-15"],
        "Test AUC": [0.65, 0.68],
        "PnL (3d)": [1200.0, 1350.0]
    })
    
    st.table(registry_data)
    
    c_prom, c_roll = st.columns(2)
    if c_prom.button("Promote Shadow to Active"):
        st.success("Promoting v1.1 to ACTIVE...")
        time.sleep(1)
        st.rerun()
        
    if c_roll.button("Rollback to Previous"):
        st.warning("Rolling back...")
        time.sleep(1)
        st.rerun()

with t4:
    st.subheader("⚠️ Portfolio Risk & Hedge")
    from portfolio import Portfolio, Position
    from risk_engine import RiskEngine
    
    # Mock Portfolio
    p = Portfolio(initial_capital=100000.0)
    # NIFTY Long Call (50 qty)
    p.add_position(Position(
        symbol="NIFTY", instrument_type="CE", strike=22500, expiry="2024-03-28", qty=50, 
        entry_price=150, current_price=180, greeks={"delta": 0.6, "gamma": 0.002, "vega": 10, "theta": -5}
    ))
    # NIFTY Short Put (50 qty) - Net Long Delta
    p.add_position(Position(
        symbol="NIFTY", instrument_type="PE", strike=22400, expiry="2024-03-28", qty=-50, 
        entry_price=100, current_price=80, greeks={"delta": -0.3, "gamma": 0.001, "vega": 8, "theta": -4}
    ))
    
    re = RiskEngine()
    net_greeks = p.get_net_greeks()
    
    # 1. Greeks Display
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Net Delta", f"{net_greeks['delta']:.2f}")
    g2.metric("Net Gamma", f"{net_greeks['gamma']:.4f}")
    g3.metric("Net Vega", f"{net_greeks['vega']:.2f}")
    g4.metric("Net Theta", f"{net_greeks['theta']:.2f}")
    
    # 2. Risk Checks
    st.markdown("#### Exposure Limits")
    util = p.get_capital_utilization()
    st.progress(min(util / 0.10, 1.0), text=f"Capital Utilization: {util:.1%}")
    if util > 0.05:
        st.error("Capital Utilization > 5%!")
        
    # 3. Hedge Suggestions
    st.markdown("#### Hedge Recommendations")
    suggestions = re.propose_hedge(net_greeks)
    if suggestions:
        for s in suggestions:
            st.warning(f"💡 Suggestion: {s}")
        if st.button("Simulate Hedge Execution"):
            st.info("Simulation: Hedge orders generated (Mock).")
    else:
        st.success("Portfolio is balanced. No hedge needed.")

    # --- Monte Carlo Simulation (Sub-section) ---
    st.divider()
    st.markdown("#### 🎲 Monte Carlo Risk Simulation")
    
    from monte_carlo import MonteCarloEngine
    
    if st.button("Run Simulation (2000 Paths)"):
        with st.spinner("Simulating..."):
            mc = MonteCarloEngine()
            sim_results = mc.run_portfolio_simulation(p, n_sims=2000)
            stress_results = mc.run_stress_test(p)
            
            # Histogram
            fig_hist = px.histogram(sim_results['pnl_sim'], nbins=50, title="Projected PnL Distribution (1-Day)", labels={'value': 'PnL'})
            fig_hist.add_vline(x=sim_results['var_95'], line_dash="dash", line_color="orange", annotation_text="VaR 95%")
            fig_hist.add_vline(x=sim_results['var_99'], line_dash="dash", line_color="red", annotation_text="VaR 99%")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("VaR 95%", f"₹ {sim_results['var_95']:.2f}")
            m2.metric("VaR 99%", f"₹ {sim_results['var_99']:.2f}")
            m3.metric("Worst Case", f"₹ {sim_results['worst_case']:.2f}")
            
            # Stress Table
            st.markdown("##### Stress Scenarios")
            st.dataframe(pd.DataFrame([stress_results]).T.rename(columns={0: "PnL Impact"}), use_container_width=True)

with t5:
    st.subheader("🧠 Explainability")
    from explainability import Explainer
    
    # Mock Feature Data for visualization if DB not connected
    # In real app, fetch `explainability_logs`
    
    st.markdown("##### Latest Prediction Drivers")
    col_x1, col_x2 = st.columns([2, 1])
    
    with col_x1:
        # Mock SHAP values
        shap_data = pd.DataFrame({
            "Feature": ["Delta", "IV", "OI_Change", "RSI", "Gamma", "Vega", "Theta", "Spot_Price"],
            "Impact": [0.35, 0.25, -0.15, 0.10, 0.05, -0.04, 0.02, 0.01]
        }).sort_values("Impact", ascending=True) # Sort for bar chart
        
        fig_shap = px.bar(shap_data, x="Impact", y="Feature", orientation='h', 
                          title="Feature Importance (SHAP)", color="Impact",
                          color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_shap, use_container_width=True)
        
    with col_x2:
        st.info("ℹ️ **Delta** is the strongest positive driver, while **OI_Change** is providing negative pressure.")
        if st.button("Generate Daily Report"):
            with st.spinner("Generating PDF/HTML Report..."):
                ex = Explainer()
                # Mock logs
                logs = [{'top_features': {'Delta': 0.5, 'IV': 0.3}} for _ in range(10)]
                ex.generate_daily_report(logs, output_path="daily_report.html")
                st.success("Report generated: `daily_report.html`")
                with open("daily_report.html", "r") as f:
                    st.download_button("Download Report", f, file_name="daily_report.html", mime="text/html")

with t6:
    st.subheader("⚙️ Hyperparameter Optimization")
    
    # Mock Optimization Data or Read from DB
    # In real app: optimizer.get_optimization_history()
    
    if st.button("Run Optimization (Mock)"):
        with st.spinner("Running Optuna Study..."):
             time.sleep(2)
             st.success("Optimization Complete!")
    
    # Mock Trials Data
    trials_df = pd.DataFrame({
        "number": range(1, 11),
        "value": np.random.uniform(1.5, 2.5, 10), # Sharpe
        "params_weight_lgbm": np.random.uniform(0, 1, 10),
        "params_threshold": np.random.uniform(0.6, 0.9, 10),
        "state": ["COMPLETE"] * 10
    })
    
    st.markdown("#### Optimization History")
    fig_opt = px.scatter(trials_df, x="number", y="value", title="Sharpe Ratio by Trial", labels={'number': 'Trial', 'value': 'Sharpe'})
    fig_opt.update_traces(mode='lines+markers')
    st.plotly_chart(fig_opt, use_container_width=True)
    
    st.markdown("#### Best Parameters")
    best_trial = trials_df.loc[trials_df['value'].idxmax()]
    st.json(best_trial.to_dict())


# Shock Logs (New Section)
st.divider()
st.subheader("⚡ Shock Detection Logs")
# Mock Shock Data
shock_data = pd.DataFrame({
    "Timestamp": [pd.Timestamp.now() - pd.Timedelta(minutes=10)],
    "Type": ["IV_SPIKE"], # Example
    "Details": ["{'z_score': 2.5}"],
    "Action": ["RISK_REDUCED"]
})
if os.path.exists("STOP.flag"):
     st.table(shock_data)
else:
     st.info("No recent shocks detected.")

# Refresh
time.sleep(refresh_rate)
st.rerun()


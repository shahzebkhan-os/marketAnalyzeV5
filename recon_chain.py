import os
import asyncio
from dotenv import load_dotenv
from growwapi import GrowwAPI
import pandas as pd
from datetime import datetime

async def reconstruct_chain():
    load_dotenv()
    api_key = os.getenv("GROWW_API_KEY")
    secret = os.getenv("GROWW_API_SECRET")
    
    sdk = GrowwAPI(api_key)
    print("Authenticating...")
    token = await asyncio.to_thread(sdk.get_access_token, api_key=api_key, secret=secret)
    session_sdk = GrowwAPI(token)
    
    print("Fetching instruments...")
    df = await asyncio.to_thread(session_sdk.get_all_instruments)
    
    # 1. Get Spot Price
    print("Fetching NIFTY Spot Price...")
    quote = await asyncio.to_thread(session_sdk.get_quote, trading_symbol="NIFTY", exchange="NSE", segment="CASH")
    spot_price = quote['last_price']
    print(f"NIFTY Spot: {spot_price}")
    
    # 2. Filter for NIFTY FNO
    fno_nifty = df[(df['underlying_symbol'] == 'NIFTY') & (df['segment'] == 'FNO')].copy()
    fno_nifty['strike_price'] = pd.to_numeric(fno_nifty['strike_price'], errors='coerce')
    
    # 3. Find Nearest Expiry
    expiries = sorted(fno_nifty['expiry_date'].dropna().unique())
    nearest_expiry = expiries[0]
    print(f"Nearest Expiry: {nearest_expiry}")
    
    # 4. Filter for Nearest Expiry
    current_chain = fno_nifty[fno_nifty['expiry_date'] == nearest_expiry]
    
    # 5. Find ATM Strike (Nearest to spot)
    unique_strikes = sorted(current_chain['strike_price'].unique())
    atm_strike = min(unique_strikes, key=lambda x: abs(x - spot_price))
    print(f"ATM Strike: {atm_strike}")
    
    # 6. Select window of strikes (e.g., 3 up, 3 down)
    idx = unique_strikes.index(atm_strike)
    start_idx = max(0, idx - 3)
    end_idx = min(len(unique_strikes), idx + 4)
    target_strikes = unique_strikes[start_idx:end_idx]
    
    print(f"Selected Strikes: {target_strikes}")
    
    # 7. Collect Symbols to Fetch
    symbols_to_fetch = []
    for strike in target_strikes:
        ce = current_chain[(current_chain['strike_price'] == strike) & (current_chain['instrument_type'] == 'CE')].iloc[0]['trading_symbol']
        pe = current_chain[(current_chain['strike_price'] == strike) & (current_chain['instrument_type'] == 'PE')].iloc[0]['trading_symbol']
        symbols_to_fetch.append(ce)
        symbols_to_fetch.append(pe)
    
    # 8. Fetch individual quotes (concurrently)
    print(f"Fetching {len(symbols_to_fetch)} quotes...")
    tasks = [asyncio.to_thread(session_sdk.get_quote, trading_symbol=s, exchange="NSE", segment="FNO") for s in symbols_to_fetch]
    results = await asyncio.gather(*tasks)
    
    # 9. Map results back
    quotes_map = dict(zip(symbols_to_fetch, results))
    
    # 10. Display Sample
    print("\n--- Reconstructed Option Chain Slice ---")
    for strike in target_strikes:
        ce_sym = current_chain[(current_chain['strike_price'] == strike) & (current_chain['instrument_type'] == 'CE')].iloc[0]['trading_symbol']
        pe_sym = current_chain[(current_chain['strike_price'] == strike) & (current_chain['instrument_type'] == 'PE')].iloc[0]['trading_symbol']
        
        ce_px = quotes_map[ce_sym]['last_price']
        pe_px = quotes_map[pe_sym]['last_price']
        ce_oi = quotes_map[ce_sym].get('open_interest', 0)
        pe_oi = quotes_map[pe_sym].get('open_interest', 0)
        
        print(f"Strike {strike:6}: CE {ce_px:8} (OI {ce_oi:8}) | PE {pe_px:8} (OI {pe_oi:8})")

if __name__ == "__main__":
    asyncio.run(reconstruct_chain())

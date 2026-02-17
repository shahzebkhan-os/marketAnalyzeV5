import os
from dotenv import load_dotenv
from growwapi import GrowwAPI
import pandas as pd

def find_nifty_index():
    load_dotenv()
    token = os.getenv("GROWW_API_KEY")
    sdk = GrowwAPI(token)
    
    df = sdk.get_all_instruments()
    
    # Check unique instrument types
    print(f"Unique Instrument Types: {df['instrument_type'].unique()}")
    
    # Search for NIFTY 50
    nifty50 = df[df['trading_symbol'].str.contains('NIFTY 50', case=False, na=False)]
    print(f"NIFTY 50 matches: {len(nifty50)}")
    if not nifty50.empty:
        print(nifty50[['exchange', 'trading_symbol', 'instrument_type', 'groww_symbol']].head())

    # Search for NIFTY (exact or close)
    nifty_exact = df[df['trading_symbol'] == 'NIFTY']
    print(f"Exact NIFTY matches: {len(nifty_exact)}")
    if not nifty_exact.empty:
        print(nifty_exact[['exchange', 'trading_symbol', 'instrument_type', 'groww_symbol']].head())

    # Check for underlying symbols
    if 'underlying_symbol' in df.columns:
        underlyings = df['underlying_symbol'].dropna().unique()
        nifty_underlyings = [u for u in underlyings if 'NIFTY' in str(u).upper()]
        print(f"Unique NIFTY underlyings: {nifty_underlyings[:10]}")

if __name__ == "__main__":
    find_nifty_index()

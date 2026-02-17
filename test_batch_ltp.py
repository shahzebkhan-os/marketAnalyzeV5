import os
from dotenv import load_dotenv
from growwapi import GrowwAPI
import pandas as pd

def test_batch_ltp():
    load_dotenv()
    api_key = os.getenv("GROWW_API_KEY")
    secret = os.getenv("GROWW_API_SECRET")
    
    sdk = GrowwAPI(api_key)
    token = sdk.get_access_token(api_key=api_key, secret=secret)
    session_sdk = GrowwAPI(token)
    
    # Get instruments to find 14 symbols
    df = session_sdk.get_all_instruments()
    fno_nifty = df[(df['underlying_symbol'] == 'NIFTY') & (df['segment'] == 'FNO')]
    symbols = fno_nifty['trading_symbol'].head(14).tolist()
    formatted = [f"NSE_{s}" for s in symbols]
    
    print(f"Testing batch get_ltp for {len(formatted)} symbols...")
    try:
        ltp = session_sdk.get_ltp(exchange_trading_symbols=tuple(formatted), segment="FNO")
        print(f"LTP Result: {ltp}")
    except Exception as e:
        print(f"LTP Error: {e}")

if __name__ == "__main__":
    test_batch_ltp()

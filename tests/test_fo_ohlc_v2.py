import os
from dotenv import load_dotenv
from growwapi import GrowwAPI

def test_fo_ohlc_batch_fixed():
    load_dotenv()
    api_key = os.getenv("GROWW_API_KEY")
    secret = os.getenv("GROWW_API_SECRET")
    
    sdk = GrowwAPI(api_key)
    token = sdk.get_access_token(api_key=api_key, secret=secret)
    session_sdk = GrowwAPI(token)
    
    # Get instruments to find real FNO NIFTY symbols
    df = session_sdk.get_all_instruments()
    fno_nifty = df[(df['underlying_symbol'] == 'NIFTY') & (df['segment'] == 'FNO')]
    symbols = fno_nifty['trading_symbol'].head(2).tolist()
    formatted = [f"NSE_{s}" for s in symbols]
    
    print(f"Testing get_ohlc for: {formatted}...")
    try:
        # Signature: (exchange_trading_symbols: Tuple[str], segment: str, timeout: Optional[int] = None)
        ohlc = session_sdk.get_ohlc(exchange_trading_symbols=tuple(formatted), segment="FNO")
        print(f"OHLC: {ohlc}")
    except Exception as e:
        print(f"OHLC Error: {e}")

if __name__ == "__main__":
    test_fo_ohlc_batch_fixed()

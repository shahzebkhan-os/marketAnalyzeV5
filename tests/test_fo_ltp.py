import os
from dotenv import load_dotenv
from growwapi import GrowwAPI

def test_fo_ltp():
    load_dotenv()
    api_key = os.getenv("GROWW_API_KEY")
    secret = os.getenv("GROWW_API_SECRET")
    
    sdk = GrowwAPI(api_key)
    print("Getting token...")
    token = sdk.get_access_token(api_key=api_key, secret=secret)
    
    session_sdk = GrowwAPI(token)
    
    # Let's find a NIFTY FNO symbol
    df = session_sdk.get_all_instruments()
    fno_nifty = df[(df['underlying_symbol'] == 'NIFTY') & (df['segment'] == 'FNO')]
    if fno_nifty.empty:
        print("No FNO NIFTY instruments found")
        return
    
    symbol = fno_nifty.iloc[0]['trading_symbol']
    print(f"Testing get_ltp for FNO symbol: {symbol}...")
    try:
        # Signature: (exchange_trading_symbols: Tuple[str], segment: str, timeout: Optional[int] = None)
        ltp = session_sdk.get_ltp(exchange_trading_symbols=(f"NSE:{symbol}",), segment="FNO")
        print(f"LTP: {ltp}")
    except Exception as e:
        print(f"LTP Error: {e}")

if __name__ == "__main__":
    test_fo_ltp()

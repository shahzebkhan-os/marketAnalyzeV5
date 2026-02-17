import os
from dotenv import load_dotenv
from growwapi import GrowwAPI

def test_fo_quote():
    load_dotenv()
    api_key = os.getenv("GROWW_API_KEY")
    secret = os.getenv("GROWW_API_SECRET")
    
    sdk = GrowwAPI(api_key)
    token = sdk.get_access_token(api_key=api_key, secret=secret)
    session_sdk = GrowwAPI(token)
    
    # Find a NIFTY FNO symbol
    df = session_sdk.get_all_instruments()
    fno_nifty = df[(df['underlying_symbol'] == 'NIFTY') & (df['segment'] == 'FNO')]
    symbol = fno_nifty.iloc[0]['trading_symbol']
    
    print(f"Testing get_quote for FNO: {symbol}...")
    try:
        quote = session_sdk.get_quote(trading_symbol=symbol, exchange="NSE", segment="FNO")
        print(f"Full Quote: {quote}")
        print(f"OI: {quote.get('open_interest')}")
    except Exception as e:
        print(f"Quote Error: {e}")

if __name__ == "__main__":
    test_fo_quote()

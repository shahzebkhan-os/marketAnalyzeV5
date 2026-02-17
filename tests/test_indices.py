import os
from dotenv import load_dotenv
from growwapi import GrowwAPI

def test_indices():
    load_dotenv()
    api_key = os.getenv("GROWW_API_KEY")
    secret = os.getenv("GROWW_API_SECRET")
    
    sdk = GrowwAPI(api_key)
    token = sdk.get_access_token(api_key=api_key, secret=secret)
    session_sdk = GrowwAPI(token)
    
    for symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
        try:
            quote = session_sdk.get_quote(trading_symbol=symbol, exchange="NSE", segment="CASH")
            print(f"{symbol} Spot: {quote.get('last_price')}")
        except Exception as e:
            print(f"{symbol} Error: {e}")

if __name__ == "__main__":
    test_indices()

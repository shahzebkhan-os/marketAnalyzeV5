import os
from dotenv import load_dotenv
from growwapi import GrowwAPI

def test_new_token_metadata():
    load_dotenv()
    api_key = os.getenv("GROWW_API_KEY")
    secret = os.getenv("GROWW_API_SECRET")
    
    sdk = GrowwAPI(api_key)
    print("Getting fresh token...")
    new_token = sdk.get_access_token(api_key=api_key, secret=secret)
    
    print(f"Using new token to fetch instruments...")
    new_sdk = GrowwAPI(new_token)
    try:
        # Check if generic metadata still works
        instruments = new_sdk.get_all_instruments()
        print(f"Instruments count: {len(instruments)}")
        
        # Check if basic LTP works (sometimes less restricted)
        ltp = new_sdk.get_ltp(exchange="NSE", trading_symbol="NIFTY")
        print(f"LTP NIFTY: {ltp}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_new_token_metadata()

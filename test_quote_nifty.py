import os
from dotenv import load_dotenv
from growwapi import GrowwAPI

def test_quote():
    load_dotenv()
    api_key = os.getenv("GROWW_API_KEY")
    secret = os.getenv("GROWW_API_SECRET")
    
    sdk = GrowwAPI(api_key)
    print("Getting token...")
    token = sdk.get_access_token(api_key=api_key, secret=secret)
    
    session_sdk = GrowwAPI(token)
    print("Testing get_quote for NIFTY on NSE...")
    try:
        quote = session_sdk.get_quote(trading_symbol="NIFTY", exchange="NSE", segment="CASH")
        print(f"Quote: {quote}")
    except Exception as e:
        print(f"Quote Error: {e}")

if __name__ == "__main__":
    test_quote()

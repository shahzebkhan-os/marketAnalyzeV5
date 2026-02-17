import os
from dotenv import load_dotenv
from growwapi import GrowwAPI

def test_new_token():
    load_dotenv()
    api_key = os.getenv("GROWW_API_KEY")
    secret = os.getenv("GROWW_API_SECRET")
    
    sdk = GrowwAPI(api_key)
    print("Getting fresh token...")
    auth_resp = sdk.get_access_token(api_key=api_key, secret=secret)
    # The SDK seems to return a string (the token) or a dict?
    # Based on response: it was a long string
    new_token = auth_resp
    
    print(f"Using new token to fetch expiries...")
    new_sdk = GrowwAPI(new_token)
    try:
        expiries = new_sdk.get_expiries(exchange="NSE", underlying_symbol="NIFTY")
        print(f"Expiries: {expiries}")
    except Exception as e:
        print(f"Expiries error: {e}")

if __name__ == "__main__":
    test_new_token()

import os
from dotenv import load_dotenv
from growwapi import GrowwAPI

def test_ltp():
    load_dotenv()
    token = os.getenv("GROWW_API_KEY")
    sdk = GrowwAPI(token)
    
    print("Fetching LTP for RELIANCE...")
    try:
        # get_ltp(exchange: str, trading_symbol: str)
        ltp = sdk.get_ltp(exchange="NSE", trading_symbol="RELIANCE")
        print(f"LTP: {ltp}")
    except Exception as e:
        print(f"LTP error: {e}")

if __name__ == "__main__":
    test_ltp()

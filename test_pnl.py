import os
from dotenv import load_dotenv
from growwapi import GrowwAPI

def test_pnl_access():
    load_dotenv()
    token = os.getenv("GROWW_API_KEY")
    sdk = GrowwAPI(token)
    
    print("Testing OHLC access for NIFTY...")
    try:
        # get_ohlc might have a different signature, let's check it first
        import inspect
        sig = inspect.signature(sdk.get_ohlc)
        print(f"get_ohlc signature: {sig}")
        
        # Try calling it
        ohlc = sdk.get_ohlc(exchange="NSE", trading_symbol="NIFTY")
        print(f"OHLC: {ohlc}")
    except Exception as e:
        print(f"OHLC error: {e}")

if __name__ == "__main__":
    test_pnl_access()

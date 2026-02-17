import asyncio
import os
from dotenv import load_dotenv
from growwapi import GrowwAPI
import structlog

# Setup logging
structlog.configure()
logger = structlog.get_logger()

load_dotenv()

async def test_sdk():
    token = os.getenv("GROWW_API_KEY")
    if not token or "your_api_key" in token:
        print("ERROR: GROWW_API_KEY not found in .env")
        return

    print(f"Using token (first 10 chars): {token[:10]}...")
    
    try:
        sdk = GrowwAPI(token)
        exchange = "NSE"
        symbol = "NIFTY"
        
        print(f"Fetching expiries for {symbol} on {exchange}...")
        expiries = sdk.get_expiries(exchange=exchange, underlying=symbol)
        print(f"Expiries: {expiries}")
        
        if expiries and isinstance(expiries, list):
            nearest_expiry = expiries[0]
            print(f"Fetching option chain for {nearest_expiry}...")
            response = sdk.get_option_chain(
                exchange=exchange,
                underlying=symbol,
                expiry_date=nearest_expiry
            )
            print("Response keys:", response.keys() if response else "None")
            print("Underlying LTP:", response.get("underlying_ltp") if response else "N/A")
            strikes = response.get("strikes", {}) if response else {}
            print(f"Number of strikes: {len(strikes)}")
            if strikes:
                first_strike = list(strikes.keys())[0]
                print(f"Sample strike ({first_strike}) data: {strikes[first_strike]}")
        else:
            print("No expiries found. Trying 'NIFTY 50'...")
            expiries = sdk.get_expiries(exchange=exchange, underlying="NIFTY 50")
            print(f"Expiries (NIFTY 50): {expiries}")

    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_sdk())

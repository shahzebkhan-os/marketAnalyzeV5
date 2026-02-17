import asyncio
import os
from dotenv import load_dotenv
from growwapi import GrowwAPI
import structlog
import json

# Setup logging
structlog.configure()
logger = structlog.get_logger()

load_dotenv()

async def test_sdk():
    token = os.getenv("GROWW_API_KEY")
    if not token or "your_api_key" in token:
        print("ERROR: GROWW_API_KEY not found in .env")
        return

    try:
        sdk = GrowwAPI(token)
        exchange = "NSE"
        symbol = "NIFTY"
        
        print(f"Fetching expiries for {symbol} on {exchange}...")
        # Correct argument is underlying_symbol
        response = sdk.get_expiries(exchange=exchange, underlying_symbol=symbol)
        print(f"Expiries response type: {type(response)}")
        print(f"Expiries response: {response}")
        
    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_sdk())

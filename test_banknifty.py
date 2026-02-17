import os
import asyncio
from dotenv import load_dotenv
from growwapi import GrowwAPI
import pandas as pd
from client import GrowwClient

async def test_banknifty():
    load_dotenv()
    async with GrowwClient() as client:
        print("Fetching BANKNIFTY option chain...")
        # BANKNIFTY is the symbol for Groww
        try:
            data = await client.fetch_option_chain("BANKNIFTY")
            print(f"BANKNIFTY Spot: {data['spot_price']}")
            chain = data['option_chain']
            print(f"Chain length: {len(chain)}")
            if chain:
                # Find strike nearest to spot
                strikes = sorted(list(set(item['strike'] for item in chain)))
                atm_strike = min(strikes, key=lambda x: abs(x - data['spot_price']))
                print(f"ATM Strike: {atm_strike}")
                
                atm_data = [item for item in chain if item['strike'] == atm_strike]
                print(f"ATM Chain Data: {atm_data}")
        except Exception as e:
            print(f"Error fetching BANKNIFTY: {e}")

if __name__ == "__main__":
    asyncio.run(test_banknifty())

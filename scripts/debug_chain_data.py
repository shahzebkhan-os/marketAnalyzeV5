import asyncio
import json
from client import GrowwClient

async def debug_data():
    async with GrowwClient() as client:
        # We'll use the web provider directly to see what actually comes back
        # Since we want to see the raw data structure
        try:
            print("Fetching NIFTY chain via GrowwClient...")
            data = await client.fetch_option_chain("NIFTY")
            if data and data.get('option_chain'):
                first_row = data['option_chain'][0]
                print(f"Sample Row: {first_row}")
                # Let's see if we can get even more raw data if we were to look at the provider
                # But fetch_option_chain already flattens it.
                # I'll check if I can modify client.py to include oi_change.
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(debug_data())

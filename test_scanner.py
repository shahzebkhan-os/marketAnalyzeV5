import asyncio
import os
from dotenv import load_dotenv
from client import GrowwClient
import pandas as pd

async def test_scanner():
    load_dotenv()
    
    print("Initializing GrowwClient...")
    try:
        async with GrowwClient() as client:
            print("Starting Market Scan 1 (Should fetch full instruments)...")
            start = asyncio.get_event_loop().time()
            results = await client.fetch_market_scan()
            end = asyncio.get_event_loop().time()
            print(f"Scan 1 took {end-start:.2f}s")
            
            print("\nStarting Market Scan 2 (Should use cache)...")
            start = asyncio.get_event_loop().time()
            results2 = await client.fetch_market_scan()
            end = asyncio.get_event_loop().time()
            print(f"Scan 2 took {end-start:.2f}s")
            
            if results:
                df = pd.DataFrame(results)
                print(f"Successfully found {len(df)} F&O underlyings.")
                print("\nTop 10 underlyings found:")
                print(df.head(10))
                
                # Test deep dive analysis logic for one candidate
                cand = df.iloc[0]['symbol']
                print(f"\nTesting Analysis for {cand}...")
                data = await client.fetch_option_chain(cand)
                chain = data.get('option_chain', [])
                if chain:
                    chain_df = pd.DataFrame(chain)
                    best = chain_df.sort_values('oi', ascending=False).iloc[0]
                    print(f"Best setup for {cand}: {best['strike']} {best['type']} (OI: {best['oi']})")
                else:
                    print(f"No chain data found for {cand}")
            else:
                print("No results returned from scanner.")
    except Exception as e:
        print(f"Scan failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_scanner())

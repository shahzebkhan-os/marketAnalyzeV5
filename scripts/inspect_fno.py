import os
from dotenv import load_dotenv
from growwapi import GrowwAPI
import pandas as pd

def inspect_fno_nifty():
    load_dotenv()
    token = os.getenv("GROWW_API_KEY")
    sdk = GrowwAPI(token)
    
    df = sdk.get_all_instruments()
    
    # Filter for NIFTY underlyings in FNO segment
    fno_nifty = df[(df['underlying_symbol'] == 'NIFTY') & (df['segment'] == 'FNO')]
    print(f"Total FNO NIFTY rows: {len(fno_nifty)}")
    
    if not fno_nifty.empty:
        print("Sample columns for FNO NIFTY:")
        print(fno_nifty[['exchange', 'trading_symbol', 'instrument_type', 'groww_symbol', 'underlying_symbol', 'underlying_exchange_token']].head(10))
        print("\nExchanges used in FNO NIFTY:", fno_nifty['exchange'].unique())

if __name__ == "__main__":
    inspect_fno_nifty()

import os
from dotenv import load_dotenv
from growwapi import GrowwAPI
import pandas as pd

def analyze_instruments():
    load_dotenv()
    token = os.getenv("GROWW_API_KEY")
    sdk = GrowwAPI(token)
    
    print("Fetching instruments...")
    df = sdk.get_all_instruments() # This returns a DataFrame
    
    print(f"Columns: {df.columns.tolist()}")
    
    # Filter for NIFTY
    # Common names: NIFTY, NIFTY 50, NIFTY BANK
    nifty_df = df[df['trading_symbol'].str.contains('NIFTY', case=False, na=False)]
    print(f"Total NIFTY related: {len(nifty_df)}")
    
    # Check unique segments for NIFTY
    if 'segment' in df.columns:
         print(f"Segments for NIFTY: {nifty_df['segment'].unique()}")
    
    # Show NIFTY indices or base symbols
    # Indices often have a specific exchange or type
    print("Sample NIFTY entries:")
    print(nifty_df[['exchange', 'trading_symbol', 'segment', 'instrument_type']].head(20))

    # Look for the specific underlying we need for option chain
    # Usually it's an index.
    indices = df[df['instrument_type'] == 'INDEX'] # Assuming instrument_type column exists
    print(f"\nIndices matching NIFTY: {indices[indices['trading_symbol'].str.contains('NIFTY', case=False, na=False)][['exchange', 'trading_symbol']]}")

if __name__ == "__main__":
    analyze_instruments()

import os
from dotenv import load_dotenv
from growwapi import GrowwAPI
import json

def list_instruments():
    load_dotenv()
    token = os.getenv("GROWW_API_KEY")
    sdk = GrowwAPI(token)
    
    print("Fetching instruments...")
    try:
        # This might return a large list or a dict
        instruments = sdk.get_all_instruments()
        print(f"Type: {type(instruments)}")
        if isinstance(instruments, list):
            print(f"Count: {len(instruments)}")
            # Show first 5
            print(f"Sample: {instruments[:5]}")
            # Search for NIFTY
            nifty_matches = [i for i in instruments if 'NIFTY' in str(i).upper()]
            print(f"NIFTY matches (count {len(nifty_matches)}): {nifty_matches[:10]}")
        else:
            print(f"Response: {instruments}")
    except Exception as e:
        print(f"Instruments error: {e}")

if __name__ == "__main__":
    list_instruments()

import os
from dotenv import load_dotenv
from growwapi import GrowwAPI

def test_equity():
    load_dotenv()
    token = os.getenv("GROWW_API_KEY")
    sdk = GrowwAPI(token)
    
    print("Fetching quote for RELIANCE...")
    try:
        # get_quote usually works for equity
        quote = sdk.get_quote(exchange="NSE", symbol="RELIANCE")
        print(f"Quote: {quote}")
    except Exception as e:
        print(f"Quote error: {e}")

if __name__ == "__main__":
    test_equity()

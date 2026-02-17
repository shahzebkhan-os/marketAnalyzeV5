import os
from dotenv import load_dotenv
from growwapi import GrowwAPI
import json

def comprehensive_test():
    load_dotenv()
    api_key = os.getenv("GROWW_API_KEY")
    secret = os.getenv("GROWW_API_SECRET")
    
    sdk = GrowwAPI(api_key)
    print("Getting fresh session token...")
    token = sdk.get_access_token(api_key=api_key, secret=secret)
    session_sdk = GrowwAPI(token)
    
    symbol = "NIFTY"
    exchange = "NSE"
    
    print(f"\n--- Testing Expiries for {symbol} ({exchange}) ---")
    try:
        expiries = session_sdk.get_expiries(exchange=exchange, underlying_symbol=symbol)
        print(f"Expiries count: {len(expiries)}")
        print(f"Nearest: {expiries[0] if expiries else 'None'}")
        
        if expiries:
            target_expiry = expiries[0]
            print(f"\n--- Testing Option Chain for {symbol} on {target_expiry} ---")
            try:
                chain = session_sdk.get_option_chain(exchange=exchange, underlying=symbol, expiry_date=target_expiry)
                print("Option Chain fetch: SUCCESS")
                # print(f"Sample Strike: {next(iter(chain.get('strike_map', {}).keys()))}")
            except Exception as e:
                print(f"Option Chain fetch: FAILED ({e})")
    except Exception as e:
        print(f"Expiries fetch: FAILED ({e})")

    print("\n--- Testing Quote for NIFTY ---")
    try:
        quote = session_sdk.get_quote(trading_symbol=symbol, exchange=exchange, segment="CASH")
        print(f"Quote last price: {quote.get('last_price')}")
    except Exception as e:
        print(f"Quote fetch: FAILED ({e})")

if __name__ == "__main__":
    comprehensive_test()

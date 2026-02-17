import os
from dotenv import load_dotenv
from growwapi import GrowwAPI

def test_ohlc_correct():
    load_dotenv()
    token = os.getenv("GROWW_API_KEY")
    sdk = GrowwAPI(token)
    
    print("Testing OHLC with correct signature...")
    try:
        # tuple of 'exchange:trading_symbol'
        symbols = ("NSE:NIFTY",)
        ohlc = sdk.get_ohlc(exchange_trading_symbols=symbols, segment="CASH")
        print(f"OHLC (CASH): {ohlc}")
        
        ohlc_fno = sdk.get_ohlc(exchange_trading_symbols=symbols, segment="FNO")
        print(f"OHLC (FNO): {ohlc_fno}")
    except Exception as e:
        print(f"OHLC error: {e}")

if __name__ == "__main__":
    test_ohlc_correct()

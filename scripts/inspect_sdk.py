import os
from dotenv import load_dotenv
from growwapi import GrowwAPI
import inspect

def inspect_sdk():
    token = os.getenv("GROWW_API_KEY", "dummy")
    sdk = GrowwAPI(token)
    
    methods = [m for m in dir(sdk) if not m.startswith('_')]
    print(f"Available methods: {methods}")
    
    for method_name in ['get_expiries', 'get_option_chain', 'get_greeks']:
        if hasattr(sdk, method_name):
            method = getattr(sdk, method_name)
            try:
                sig = inspect.signature(method)
                print(f"Signature for {method_name}: {sig}")
            except Exception as e:
                print(f"Could not get signature for {method_name}: {e}")

if __name__ == "__main__":
    inspect_sdk()

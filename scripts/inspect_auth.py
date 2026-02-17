import os
from dotenv import load_dotenv
from growwapi import GrowwAPI
import inspect

def inspect_auth():
    sdk = GrowwAPI("dummy")
    if hasattr(sdk, 'get_access_token'):
        method = getattr(sdk, 'get_access_token')
        try:
            sig = inspect.signature(method)
            print(f"Signature for get_access_token: {sig}")
        except Exception as e:
            print(f"Could not get signature for get_access_token: {e}")
    else:
        print("SDK has no get_access_token method")

if __name__ == "__main__":
    inspect_auth()

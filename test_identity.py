import os
from dotenv import load_dotenv
from growwapi import GrowwAPI
import json

def test_full_identity():
    load_dotenv()
    api_key = os.getenv("GROWW_API_KEY")
    secret = os.getenv("GROWW_API_SECRET")
    
    sdk = GrowwAPI(api_key)
    print("Getting session token via exchange...")
    token = sdk.get_access_token(api_key=api_key, secret=secret)
    
    session_sdk = GrowwAPI(token)
    
    print("\n--- User Profile ---")
    try:
        # Check if get_user_profile exists and signature
        import inspect
        if hasattr(session_sdk, 'get_user_profile'):
            profile = session_sdk.get_user_profile()
            print(f"Profile: {profile}")
        else:
            print("SDK has no get_user_profile method")
    except Exception as e:
        print(f"Profile Error: {e}")

    print("\n--- Account Funds ---")
    try:
        if hasattr(session_sdk, 'get_available_margin_details'):
            margin = session_sdk.get_available_margin_details()
            print(f"Margin: {margin}")
    except Exception as e:
        print(f"Margin Error: {e}")

if __name__ == "__main__":
    test_full_identity()

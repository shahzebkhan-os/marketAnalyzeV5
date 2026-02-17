import os
from dotenv import load_dotenv
from growwapi import GrowwAPI

def test_token_exchange():
    load_dotenv()
    api_key = os.getenv("GROWW_API_KEY")
    secret = os.getenv("GROWW_API_SECRET")
    
    print(f"Attempting to exchange key/secret for token...")
    # Initialize with dummy or existing key
    sdk = GrowwAPI(api_key)
    
    try:
        # Signature: (api_key: str, totp: Optional[str] = None, secret: Optional[str] = None)
        response = sdk.get_access_token(api_key=api_key, secret=secret)
        print(f"Token exchange response: {response}")
    except Exception as e:
        print(f"Token exchange error: {e}")

if __name__ == "__main__":
    test_token_exchange()

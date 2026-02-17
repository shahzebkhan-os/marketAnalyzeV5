import os
from dotenv import load_dotenv
from growwapi import GrowwAPI

def verify_token():
    load_dotenv()
    token = os.getenv("GROWW_API_KEY")
    sdk = GrowwAPI(token)
    
    print("Checking user profile...")
    try:
        profile = sdk.get_user_profile()
        print(f"Profile: {profile}")
    except Exception as e:
        print(f"Profile error: {e}")

if __name__ == "__main__":
    verify_token()

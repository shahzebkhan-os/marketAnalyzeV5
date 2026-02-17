import os
import base64
import json
from dotenv import load_dotenv

def decode_token():
    load_dotenv()
    token = os.getenv("GROWW_API_KEY")
    if not token or "." not in token:
        print("Invalid token format")
        return
    
    parts = token.split(".")
    if len(parts) >= 2:
        payload = parts[1]
        # Fix padding
        payload += "=" * ((4 - len(payload) % 4) % 4)
        decoded = base64.b64decode(payload).decode("utf-8")
        print(f"Decoded Payload:\n{json.dumps(json.loads(decoded), indent=2)}")

if __name__ == "__main__":
    decode_token()

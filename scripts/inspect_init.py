from growwapi import GrowwAPI
import inspect

def inspect_init():
    try:
        sig = inspect.signature(GrowwAPI.__init__)
        print(f"Signature for GrowwAPI.__init__: {sig}")
    except Exception as e:
        print(f"Could not get signature for GrowwAPI.__init__: {e}")

if __name__ == "__main__":
    inspect_init()

import os
from cryptography.fernet import Fernet
import argparse

def generate_key():
    """Generates a key and saves it into a file"""
    key = Fernet.generate_key()
    with open("secret.key", "wb") as key_file:
        key_file.write(key)
    print("Key generated and saved to secret.key")

def load_key():
    """Loads the key from the current directory named `secret.key`"""
    return open("secret.key", "rb").read()

def encrypt_file(filename):
    """Encrypts a file"""
    key = load_key()
    f = Fernet(key)
    with open(filename, "rb") as file:
        file_data = file.read()
    encrypted_data = f.encrypt(file_data)
    with open(filename + ".enc", "wb") as file:
        file.write(encrypted_data)
    print(f"Encrypted {filename} to {filename}.enc")

def decrypt_file(filename):
    """Decrypts a file"""
    key = load_key()
    f = Fernet(key)
    with open(filename, "rb") as file:
        encrypted_data = file.read()
    decrypted_data = f.decrypt(encrypted_data)
    with open(filename.replace(".enc", ""), "wb") as file:
        file.write(decrypted_data)
    print(f"Decrypted {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Secrets Manager")
    parser.add_argument("action", choices=["generate-key", "encrypt", "decrypt"])
    parser.add_argument("--file", help="File to encrypt/decrypt")
    
    args = parser.parse_args()
    
    if args.action == "generate-key":
        generate_key()
    elif args.action == "encrypt":
        if not args.file:
            print("Please provide --file")
        else:
            encrypt_file(args.file)
    elif args.action == "decrypt":
        if not args.file:
            print("Please provide --file")
        else:
            decrypt_file(args.file)

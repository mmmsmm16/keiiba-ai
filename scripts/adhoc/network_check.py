
import socket
import os
import time

def check_connection(host, port, timeout=5):
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        print(f"SUCCESS: Connected to {host}:{port}")
        return True
    except Exception as e:
        print(f"FAILURE: Could not connect to {host}:{port}. Error: {e}")
        return False

if __name__ == "__main__":
    host = os.environ.get("POSTGRES_HOST", "db")
    port = 5432
    print(f"Checking connection to {host}:{port}...")
    check_connection(host, port)

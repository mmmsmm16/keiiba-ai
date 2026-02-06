import sys
import os
import logging

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.preprocessing.loader import JraVanDataLoader

logging.basicConfig(level=logging.INFO)

def test_load():
    loader = JraVanDataLoader()
    try:
        print("Testing JraVanDataLoader.load()...")
        df = loader.load(limit=10, history_start_date="2024-01-01")
        print(f"Success! Loaded {len(df)} rows.")
        print(df[['race_id', 'title', 'grade_code', 'kyoso_joken_code']].head())
    except Exception as e:
        print(f"Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load()


import sys
import os
import logging
import pandas as pd
from sqlalchemy import create_engine

# Path setup
sys.path.append(os.path.join(os.getcwd(), 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("--- Debug Start ---")
    
    # 1. Check bloodline_stats import
    print("Checking bloodline_stats import...")
    try:
        from preprocessing.features import bloodline_stats
        print("SUCCESS: bloodline_stats imported.")
    except Exception as e:
        print(f"FAILURE: bloodline_stats import failed: {e}")

    # 2. Check Loader (bms_id)
    print("Checking Loader...")
    try:
        from preprocessing.loader import JraVanDataLoader
        loader = JraVanDataLoader()
        
        # Limit 100
        df = loader.load(limit=100, jra_only=True, history_start_date="2024-01-01")
        print(f"Loaded {len(df)} rows.")
        if 'bms_id' in df.columns:
            print(f"SUCCESS: bms_id found. Sample: {df['bms_id'].head().tolist()}")
        else:
            print("FAILURE: bms_id NOT found.")
            
    except Exception as e:
        print(f"FAILURE: Loader failed: {e}")

if __name__ == "__main__":
    main()

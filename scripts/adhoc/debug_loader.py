import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.preprocessing.loader import JraVanDataLoader

def main():
    print("Initializing loader...")
    loader = JraVanDataLoader()
    
    print("Loading small chunk (limit=100)...")
    # Load recent data to ensure tables are populated
    df = loader.load(history_start_date="2025-01-01", limit=100)
    
    print(f"Loaded {len(df)} rows.")
    if df.empty:
        print("Empty dataframe!")
        return

    check_cols = ['sex', 'surface', 'odds_10min_str', 'odds_win_str', 'state']
    
    for c in check_cols:
        if c in df.columns:
            print(f"\n[{c}]")
            print(f"Null Count: {df[c].isnull().sum()}")
            print(f"Sample: {df[c].head(5).tolist()}")
            print(f"Dtype: {df[c].dtype}")
        else:
            print(f"\n[{c}] NOT FOUND in loader output.")

if __name__ == "__main__":
    main()

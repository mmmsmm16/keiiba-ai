
import pandas as pd
import sys
import os
import logging

# Add workspace
sys.path.append('/workspace')
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.features.odds_fluctuation import compute_odds_fluctuation

# Setup Logging
logging.basicConfig(level=logging.INFO)

def main():
    print("Loading 1 day of data...")
    loader = JraVanDataLoader()
    # 2024-01-06 (Saturday)
    df = loader.load(history_start_date='2024-01-01', end_date='2024-01-06', skip_odds=False)
    
    if df.empty:
        print("No data found.")
        return
        
    print(f"Loaded {len(df)} rows.")
    print(f"Sample Race ID (Loader): {df['race_id'].iloc[0]}")
    
    print("Running compute_odds_fluctuation...")
    df_features = compute_odds_fluctuation(df)
    
    if df_features.empty:
        print("Result is Empty!")
    else:
        print(f"Result Rows: {len(df_features)}")
        print("Columns:", df_features.columns.tolist())
        print("Sample:")
        print(df_features.head())
        
        # Check odds_10min specifically
        n_valid = df_features['odds_10min'].notna().sum()
        print(f"Valid Odds 10min: {n_valid} / {len(df_features)}")

if __name__ == "__main__":
    main()

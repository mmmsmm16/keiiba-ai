
import pandas as pd
import numpy as np
import os

DATA_PATH = "data/processed/preprocessed_data_v12.parquet"

def check_features():
    if not os.path.exists(DATA_PATH):
        print(f"File not found: {DATA_PATH}")
        return

    print(f"Loading {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Shape: {df.shape}")
    
    # Check new features
    new_cols = [
        'mining_kubun', 'yoso_juni', 'yoso_time_diff', 'yoso_rank_diff',
        'horse_going_count', 'horse_going_win_rate', 'horse_going_top3_rate', 'is_proven_mudder'
    ]
    
    print("\n--- New Features Stats ---")
    for c in new_cols:
        if c in df.columns:
            print(f"\nFeature: {c}")
            print(df[c].describe())
            print(f"Null count: {df[c].isnull().sum()}")
            if df[c].dtype == 'object' or df[c].nunique() < 20:
                print("Value Counts:")
                print(df[c].value_counts().head(10))
        else:
            print(f"❌ Missing: {c}")

    # Check for NaN in critical columns
    print("\n--- Critical Check ---")
    print(f"Nulls in rank: {df['rank'].isnull().sum()}")
    print(f"Nulls in date: {df['date'].isnull().sum()}")

if __name__ == "__main__":
    check_features()

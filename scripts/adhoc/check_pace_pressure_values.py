
import pandas as pd
import os
import glob

def check_values():
    path = "data/features/pace_pressure_stats.parquet"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    df = pd.read_parquet(path)
    print(f"Loaded {path}. Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    
    feature_cols = [c for c in df.columns if c not in ['race_id', 'horse_number', 'horse_id']]
    
    for col in feature_cols:
        print(f"\n--- {col} ---")
        print(df[col].describe())
        print(f"NaN count: {df[col].isna().sum()}")
        print(f"Zero count: {(df[col] == 0).sum()}")

if __name__ == "__main__":
    check_values()

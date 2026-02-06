
import pandas as pd
import os

def check_parquet():
    feat_path = "data/temp_t2/T2_features.parquet"
    if os.path.exists(feat_path):
        df = pd.read_parquet(feat_path)
        print(f"Features: {df.shape}")
        if 'date' in df.columns:
            print(f"Date Range: {df['date'].min()} ~ {df['date'].max()}")
        elif 'race_id' in df.columns:
             # Try to infer from race_id just in case
             print("No date col, checking race_id...")
             print(df['race_id'].head())
    else:
        print("Features parquet not found.")

    tgt_path = "data/temp_t2/T2_targets.parquet"
    if os.path.exists(tgt_path):
        df = pd.read_parquet(tgt_path)
        print(f"Targets: {df.shape}")
        if 'date' in df.columns:
             print(f"Date Range: {df['date'].min()} ~ {df['date'].max()}")
    else:
        print("Targets parquet not found.")

if __name__ == "__main__":
    check_parquet()

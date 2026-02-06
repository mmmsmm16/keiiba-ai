
import pandas as pd
import os

PATH = "data/features_t2/rebuild_v12/pace_stats.parquet"

def check():
    if not os.path.exists(PATH):
        print("File not found.")
        return
        
    try:
        df = pd.read_parquet(PATH)
        print(f"Columns: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")
        print(df.head())
    except Exception as e:
        print(f"Error reading parquet: {e}")

if __name__ == "__main__":
    check()

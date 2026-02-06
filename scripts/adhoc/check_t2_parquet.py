
import pandas as pd
import os

def check_file(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    try:
        df = pd.read_parquet(path)
        print(f"--- {path} ---")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()[:20]} ...")
        if 'date' in df.columns:
            print("Contains 'date'")
        else:
            print("Does NOT contain 'date'")
            # Check for date-like
            print([c for c in df.columns if 'date' in c])
    except Exception as e:
        print(f"Error reading {path}: {e}")

def main():
    check_file("data/temp_t2/T2_features.parquet")
    check_file("data/temp_t2/T2_targets.parquet")

if __name__ == "__main__":
    main()

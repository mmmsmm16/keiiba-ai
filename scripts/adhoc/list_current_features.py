
import pandas as pd
import os

def list_features():
    path = "data/processed/preprocessed_data_v11.parquet"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    # Use PyArrow to read schema only (much faster than reading data)
    try:
        df = pd.read_parquet(path)
        cols = df.columns.tolist()
        print(f"Total Columns: {len(cols)}")
        for c in cols:
            print(c)
    except Exception as e:
        print(f"Error reading parquet: {e}")

if __name__ == "__main__":
    list_features()

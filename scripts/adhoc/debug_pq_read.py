import pandas as pd
import os
import sys

def main():
    path = 'data/processed/preprocessed_data.parquet'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print("Attempting to read parquet...")
    try:
        df = pd.read_parquet(path)
        print(f"Success! Shape: {df.shape}")
        # print(df.dtypes.head())
        with open('cols.txt', 'w') as f:
            f.write('\n'.join(list(df.columns)))
        print("Columns written to cols.txt")
    except Exception as e:
        print(f"Read failed: {e}")
        
    print("Checking engines...")
    try:
        import pyarrow
        print(f"pyarrow version: {pyarrow.__version__}")
    except:
        print("pyarrow not installed")
        
    try:
        import fastparquet
        print(f"fastparquet version: {fastparquet.__version__}")
    except:
        print("fastparquet not installed")

if __name__ == "__main__":
    main()

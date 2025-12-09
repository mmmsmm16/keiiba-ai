
import pandas as pd
import os

path = '/workspace/data/processed/preprocessed_data.parquet'
if os.path.exists(path):
    df = pd.read_parquet(path, columns=['date'])
    df['date'] = pd.to_datetime(df['date'])
    print(f"File: {path}")
    print(f"Shape: {df.shape}")
    print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique Years: {df['date'].dt.year.unique()}")
else:
    print(f"File not found: {path}")

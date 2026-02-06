
import pandas as pd
import os

path = "data/temp_t2/T2_predictions_2024_2025.parquet"
if os.path.exists(path):
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    print(f"File: {path}")
    print(f"Min Date: {df['date'].min()}")
    print(f"Max Date: {df['date'].max()}")
    print(f"Row count: {len(df)}")
    print(f"2024 rows: {len(df[df['date'].dt.year == 2024])}")
    print(f"2025 rows: {len(df[df['date'].dt.year == 2025])}")
else:
    print(f"File not found: {path}")

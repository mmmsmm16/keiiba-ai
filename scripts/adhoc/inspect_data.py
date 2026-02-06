import pandas as pd
import glob
import os

files = [
    'data/derived/preprocessed_with_prob_v12.parquet',
]

for f in files:
    if os.path.exists(f):
        print(f"=== {f} ===")
        try:
            df = pd.read_parquet(f)
            print(f"Shape: {df.shape}")
            # print(f"Columns: {df.columns.tolist()[:10]} ...")
            print(f"All Columns: {df.columns.tolist()}")
            
            # Check for segment columns aliases
            potential_targets = ['rank', 'order', 'finish_pos', '着順', 'order_of_finish']
            found_targets = [c for c in df.columns if c in potential_targets]
            print(f"Found target candidates: {found_targets}")

            potential_n = ['n_horses', 'n_entries', 'entry_count', 'tosu', 'head_count']
            found_n = [c for c in df.columns if c in potential_n]
            print(f"Found n_entries candidates: {found_n}")

            
            # Check distinct years
            print("Years present:", sorted(df['date'].dt.year.unique()))
        except Exception as e:
            print(f"Error reading {f}: {e}")

import pandas as pd
import os

path = "data/features_t2/incremental_cache/base_attributes.parquet"

if not os.path.exists(path):
    print(f"Cache file not found: {path}")
else:
    try:
        df = pd.read_parquet(path)
        print(f"Loaded {path}. Shape: {df.shape}")
        
        target_cols = ['sex', 'surface']
        for c in target_cols:
            if c in df.columns:
                print(f"\n[{c}]")
                print(f"Null Count: {df[c].isnull().sum()}")
                print(f"Sample: {df[c].dropna().head(5).tolist()}")
                print(f"Dtype: {df[c].dtype}")
            else:
                print(f"\n[{c}] NOT FOUND in cache.")
                
    except Exception as e:
        print(f"Error: {e}")

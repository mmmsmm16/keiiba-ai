
import pandas as pd
import numpy as np

try:
    df = pd.read_parquet('data/processed/preprocessed_data.parquet')
    print("Columns:", df.columns.tolist())
    print("Index:", df.index.name)
    print("Race ID in columns?", 'race_id' in df.columns)
    
    if 'race_id' not in df.columns:
        print("Resetting index...")
        df = df.reset_index()
        print("Race ID in columns now?", 'race_id' in df.columns)
        
    # Check slicing
    train_df = df.head(100).copy()
    print("Train DF Race ID?", 'race_id' in train_df.columns)
    
except Exception as e:
    print(e)


import pandas as pd
import numpy as np

def run():
    # Mock DF
    df = pd.DataFrame({
        'race_id': ['1', '1', '2', '2'],
        'year': [2020, 2020, 2024, 2024],
        'val': [1, 2, 3, 4]
    })
    print("Original Cols:", df.columns.tolist())
    
    # Logic from script
    train_df = df[df['year'].isin([2020])].copy()
    valid_df = df[df['year'] == 2024].copy()
    
    print("Train Cols:", train_df.columns.tolist())
    
    # Feature cols logic reuse
    feature_cols = ['val', 'missing_col']
    if feature_cols:
        missing = set(feature_cols) - set(train_df.columns)
        for c in missing: train_df[c] = 0
        for c in missing: valid_df[c] = 0
        
    print("Train Cols after fill:", train_df.columns.tolist())
    
    # Score logic
    train_df['score'] = [0.1, 0.2]
    
    # Groupby logic
    try:
        print("Grouping by string 'race_id'...")
        for k, g in train_df.groupby('race_id'):
            pass
        print("Success.")
    except Exception as e:
        print("Failed:", e)
        
    try:
        print("Grouping by Series train_df['race_id']...")
        for k, g in train_df.groupby(train_df['race_id']):
            pass
        print("Success.")
    except Exception as e:
        print("Failed:", e)
        
run()

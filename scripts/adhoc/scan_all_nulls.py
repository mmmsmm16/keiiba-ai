import pandas as pd
import numpy as np
import os

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"

def main():
    if not os.path.exists(DATA_PATH):
        print(f"File not found: {DATA_PATH}")
        return

    print("Loading data...")
    # Load all columns
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} records.")
    
    null_threshold = 0.99
    constant_threshold = 0.99 # if one value dominates 99% (except null)
    
    broken_features = []
    
    print("\nScanning for broken features (Null > 99% or Constant)...")
    
    for col in df.columns:
        if col in ['race_id', 'horse_number', 'horse_id', 'date', 'rank', 'jockey_id', 'trainer_id', 'sire_id', 'owner_id']:
            continue
            
        # Null check
        null_rate = df[col].isnull().mean()
        if null_rate > null_threshold:
            print(f"[NULL] {col}: {null_rate:.2%}")
            broken_features.append(col)
            continue
            
        # Constant check (excluding nulls)
        if df[col].nunique() <= 1:
             # Check if it really is constant
             valid_vals = df[col].dropna()
             if len(valid_vals) > 0:
                 if valid_vals.nunique() == 1:
                     print(f"[CONST] {col}: Value={valid_vals.iloc[0]} (Null={null_rate:.2%})")
                     broken_features.append(col)
             else:
                 # All null
                 pass 
        else:
             # Dominant value check
             valid_vals = df[col].dropna()
             if len(valid_vals) > 0:
                 top_val_rate = valid_vals.value_counts(normalize=True).iloc[0]
                 if top_val_rate > constant_threshold:
                     print(f"[DOMINANT] {col}: TopVal={valid_vals.value_counts().index[0]} Rate={top_val_rate:.2%} (Null={null_rate:.2%})")
                     # Dominant is not broken per se (sparse features), but worth noting if unexpected.
                     # We won't add to broken_features unless it's strictly constant or null.
                     pass

    if not broken_features:
        print("\nSUCCESS: No broken features (100% Null or Constant) found!")
    else:
        print(f"\nFound {len(broken_features)} broken features.")

if __name__ == "__main__":
    main()

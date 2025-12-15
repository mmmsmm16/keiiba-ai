
import pickle
import pandas as pd
import os
import sys

# Load dataset
dataset_path = 'data/processed/lgbm_datasets_v10_leakfix.pkl'
print(f"Loading {dataset_path}...")

with open(dataset_path, 'rb') as f:
    data = pickle.load(f)

X_train = data['train']['X']
print("X_train columns:", X_train.columns[:20])

# Check typical ID columns
id_cols = ['horse_id', 'jockey_id', 'trainer_id', 'sire_id']
for col in id_cols:
    if col in X_train.columns:
        print(f"\n[{col}]")
        print(f"Min: {X_train[col].min()}")
        print(f"Max: {X_train[col].max()}")
        print(f"Sample: {X_train[col].head().tolist()}")
        
        # Check if raw ID (usually numeric string or large float)
        # JVD IDs: string mostly, or huge int
        # LabelEncoded: 0 to N
        if X_train[col].max() < 100000:
            print("=> Likely LabelEncoded")
        else:
            print("=> Likely Raw ID")
    else:
        print(f"\n[{col}] Not found in X_train")

# Check if label_encoders.pkl exists
enc_path = 'data/processed/label_encoders.pkl'
if os.path.exists(enc_path):
    print(f"\nFound {enc_path}")
else:
    print(f"\n{enc_path} NOT found")

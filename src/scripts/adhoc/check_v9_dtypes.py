import pickle
import pandas as pd
import numpy as np

with open('experiments/v12_tabnet_revival/data/lgbm_datasets.pkl', 'rb') as f:
    datasets = pickle.load(f)

X = datasets['train']['X']
print("Checking for object/category columns...")

for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
        print(f"Column: {col}, Dtype: {X[col].dtype}")
        print(f"Sample values: {X[col].unique()[:5]}")
        
print("\nChecking specifically for 'mile' value...")
for col in X.columns:
    try:
        if X[col].astype(str).str.contains('mile').any():
             print(f"Found 'mile' in column: {col}")
    except: pass

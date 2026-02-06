"""Debug script to investigate AUC=1.0 issue"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"

print("Loading data...")
df = pd.read_parquet(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

print(f"Total records: {len(df)}")
print(f"Columns: {len(df.columns)}")

# Check for target leakage columns
suspicious_cols = [c for c in df.columns if any(x in c.lower() for x in ['win', 'rank', 'place', 'top', 'finish', 'result'])]
print(f"\nSuspicious columns (potential leakage): {len(suspicious_cols)}")
for col in suspicious_cols:
    print(f"  - {col}")

# Check train/val split
df_train = df[(df['year'] >= 2019) & (df['year'] <= 2022)]
df_val = df[df['year'] == 2023]
print(f"\nTrain records: {len(df_train)} ({df_train['year'].min()}-{df_train['year'].max()})")
print(f"Val records: {len(df_val)} ({df_val['year'].min()}-{df_val['year'].max()})")

# Check for overlapping race_ids
train_races = set(df_train['race_id'].unique())
val_races = set(df_val['race_id'].unique())
overlap = train_races.intersection(val_races)
print(f"\nOverlapping race_ids: {len(overlap)}")

# Check target distribution
y_train = (df_train['rank'] == 1).astype(int)
y_val = (df_val['rank'] == 1).astype(int)
print(f"\nTrain target: {y_train.mean():.3%} positive ({y_train.sum()} wins)")
print(f"Val target: {y_val.mean():.3%} positive ({y_val.sum()} wins)")

# Check if is_win exists and matches rank==1
if 'is_win' in df.columns:
    print(f"\nis_win column exists:")
    print(f"  Train: is_win matches (rank==1): {(df_train['is_win'] == (df_train['rank']==1)).all()}")
    print(f"  Val: is_win matches (rank==1): {(df_val['is_win'] == (df_val['rank']==1)).all()}")
    
    # This would be a leakage if we use is_win as feature!
    print(f"\nWARNING: If 'is_win' is included as a feature, this is data leakage!")

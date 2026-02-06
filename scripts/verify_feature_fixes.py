#!/usr/bin/env python3
"""
Verification script to check if feature fixes worked.
"""
import sys
sys.path.insert(0, '/workspace')

import pandas as pd
import numpy as np
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline

RACE_ID = "202606010910"
DATE = "2026-01-25"

loader = JraVanDataLoader()

# Get horses
q = f"""SELECT DISTINCT ketto_toroku_bango FROM jvd_se 
WHERE kaisai_nen='2026' AND keibajo_code='06' 
AND kaisai_kai='01' AND kaisai_nichime='09' AND race_bango='10' """
horses = pd.read_sql(q, loader.engine)['ketto_toroku_bango'].astype(str).str.strip().tolist()
print(f"Horses: {len(horses)}")

# Load data
df_raw = loader.load_for_horses(horses, DATE, '2016-01-01', False)
print(f"Total rows: {len(df_raw)}")

# Preprocessing
numeric_cols = ['time', 'last_3f', 'rank', 'weight', 'weight_diff', 'impost']
for col in numeric_cols:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

if 'time' in df_raw.columns:
    min_times = df_raw.groupby('race_id')['time'].transform('min')
    df_raw['time_diff'] = (df_raw['time'] - min_times).fillna(0)

df_raw['race_id'] = df_raw['race_id'].astype(str)
df_raw['horse_number'] = pd.to_numeric(df_raw['horse_number'], errors='coerce').fillna(0).astype(int)

# Feature calculation
cache_dir = '/workspace/data/features'
pipeline = FeaturePipeline(cache_dir=cache_dir)
blocks = list(pipeline.registry.keys())
df_features = pipeline.load_features(df_raw, blocks, force=True)

# Extract target race
df_today = df_features[df_features['race_id'] == RACE_ID].copy()

print(f"\n=== Target Race ({RACE_ID}) Features ===")
print(f"Rows: {len(df_today)}")

# Check key columns
key_cols = ['horse_elo', 'jockey_win_rate_30d', 'trainer_win_rate_30d', 'last_nige_rate', 'avg_pci']
for col in key_cols:
    if col in df_today.columns:
        nunique = df_today[col].nunique()
        min_val = df_today[col].min()
        max_val = df_today[col].max()
        status = "✅ Variance" if nunique > 1 else "❌ Constant"
        print(f"{col}: {status} (nunique={nunique}, min={min_val:.3f}, max={max_val:.3f})")
    else:
        print(f"{col}: ❌ MISSING")

# Count constant columns
numeric_df = df_today.select_dtypes(include=[np.number])
constant_cols = [c for c in numeric_df.columns if numeric_df[c].nunique() <= 1]
print(f"\n=== Summary ===")
print(f"Total numeric columns: {len(numeric_df.columns)}")
print(f"Constant columns: {len(constant_cols)}")
print(f"Variable columns: {len(numeric_df.columns) - len(constant_cols)}")

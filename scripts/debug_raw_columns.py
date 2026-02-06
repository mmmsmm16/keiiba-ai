#!/usr/bin/env python3
"""
Debug script to check raw column values in JIT mode.
"""
import sys
sys.path.insert(0, '/workspace')

import pandas as pd
from src.preprocessing.loader import JraVanDataLoader

loader = JraVanDataLoader()

# Get horses for target race
q = """SELECT DISTINCT ketto_toroku_bango FROM jvd_se 
WHERE kaisai_nen='2026' AND keibajo_code='06' 
AND kaisai_kai='01' AND kaisai_nichime='09' AND race_bango='10' """
hdf = pd.read_sql(q, loader.engine)
horses = hdf['ketto_toroku_bango'].astype(str).str.strip().tolist()
print(f"Horses: {len(horses)}")

# Load with JIT mode
df = loader.load_for_horses(horses, '2026-01-25', '2016-01-01', False)
print(f"Total rows: {len(df)}")

# Check past data only
df_hist = df[df['race_id'] != '202606010910']
print(f"History rows: {len(df_hist)}")

print("\n=== Critical Column Check ===")
for c in ['rank', 'time', 'last_3f', 'pass_1', 'rank_str']:
    if c in df_hist.columns:
        nn = df_hist[c].notna().sum()
        pct = nn / len(df_hist) * 100 if len(df_hist) > 0 else 0
        print(f"{c}: {nn}/{len(df_hist)} non-null ({pct:.1f}%)")
        if df_hist[c].dtype in ['float64', 'int64', 'float32', 'int32']:
            print(f"   min={df_hist[c].min()}, max={df_hist[c].max()}, mean={df_hist[c].mean():.2f}")
        else:
            samples = df_hist[c].dropna().head(5).tolist()
            print(f"   sample: {samples}")
    else:
        print(f"{c}: ❌ MISSING")

# Sample horse data
print("\n=== Sample Horse Data ===")
sample_horse = horses[0]
df_sample = df_hist[df_hist['horse_id'] == sample_horse].sort_values('date')
print(f"Horse {sample_horse}: {len(df_sample)} past runs")
if len(df_sample) > 0:
    print(df_sample[['date', 'rank', 'time', 'last_3f', 'pass_1']].tail(5).to_string())

#!/usr/bin/env python3
"""
Debug track_variant calculation.
"""
import sys
sys.path.insert(0, '/workspace')

import pandas as pd
import numpy as np
from src.preprocessing.loader import JraVanDataLoader

RACE_ID = "202606010910"
DATE = "2026-01-25"

loader = JraVanDataLoader()

# Get horses
q = f"""SELECT DISTINCT ketto_toroku_bango FROM jvd_se 
WHERE kaisai_nen='2026' AND keibajo_code='06' 
AND kaisai_kai='01' AND kaisai_nichime='09' AND race_bango='10' """
horses = pd.read_sql(q, loader.engine)['ketto_toroku_bango'].astype(str).str.strip().tolist()

# Load data
df_raw = loader.load_for_horses(horses, DATE, '2016-01-01', False)

# Simulate track_variant
df_sorted = df_raw.sort_values('date').copy()
mask_top3 = (df_sorted['rank'] <= 3) & (df_sorted['time'] > 0)
df_top3 = df_sorted[mask_top3].groupby(['race_id', 'date', 'venue', 'distance', 'surface'])['time'].mean().reset_index()
df_top3.rename(columns={'time': 'race_avg_time'}, inplace=True)
df_top3 = df_top3.sort_values('date')

print(f"=== track_variant Group Analysis ===")
print(f"Total races with top3 data: {len(df_top3)}")

grp = df_top3.groupby(['venue', 'surface', 'distance'])
counts = grp.size()
print(f"\nGroup sizes (top 20):")
print(counts.sort_values(ascending=False).head(20))

# Check groups with size >= 2
large_groups = counts[counts >= 2]
print(f"\nGroups with size >= 2: {len(large_groups)}")

df_top3['rolling_time_50'] = grp['race_avg_time'].transform(lambda x: x.shift(1).rolling(50, min_periods=2).mean())
df_top3['longterm_time'] = grp['race_avg_time'].transform(lambda x: x.shift(1).expanding(min_periods=2).mean())
df_top3['track_variant'] = df_top3['rolling_time_50'] - df_top3['longterm_time']

print(f"\ntrack_variant non-null: {df_top3['track_variant'].notnull().sum()}")
if df_top3['track_variant'].notnull().sum() > 0:
    print(df_top3[df_top3['track_variant'].notnull()][['venue', 'surface', 'distance', 'track_variant']].head())
else:
    print("NO non-null track_variant values found!")

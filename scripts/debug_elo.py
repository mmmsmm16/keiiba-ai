#!/usr/bin/env python3
"""
Deep debug: Track ELO calculation step by step.
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
horses = pd.read_sql(q, loader.engine)['ketto_toroku_bango'].astype(str).str.strip().tolist()[:2]  # Just 2 horses
print(f"Sample Horses: {horses}")

# Load data 
df_raw = loader.load_for_horses(horses, DATE, '2016-01-01', False)
print(f"Total rows: {len(df_raw)}")

# Simulate ELO calculation with debugging
df_sorted = df_raw.sort_values(['date', 'race_id']).copy()
df_sorted = df_sorted.reset_index(drop=True)

K = 32
INITIAL_RATING = 1500
current_ratings = {}
res_buffer = np.zeros(len(df_sorted))

print("\n=== ELO Calculation Debug ===")
race_count = 0

for rid, grp_df in df_sorted.groupby('race_id', sort=False):
    h_ids = grp_df['horse_id'].values
    ranks = grp_df['rank'].values
    indices = grp_df.index.values
    
    # Get Current Ratings
    rs = [current_ratings.get(h, INITIAL_RATING) for h in h_ids]
    
    # Assign to Result (Pre-Race Rating)
    res_buffer[indices] = rs
    
    # Debug first 5 races
    if race_count < 5:
        print(f"\nRace {rid}:")
        print(f"  Horses: {list(h_ids)[:5]}")
        print(f"  Ranks: {list(ranks)[:5]}")
        print(f"  Pre-Ratings: {rs[:5]}")
    
    # Update Logic - same as rating_elo.py
    valid_mask = (~np.isnan(ranks)) & (ranks > 0)
    if valid_mask.sum() < 2:
        if race_count < 5:
            print(f"  Skipped: only {valid_mask.sum()} valid horses")
        race_count += 1
        continue
        
    valid_rs = np.array(rs)[valid_mask]
    valid_ranks = ranks[valid_mask]
    valid_hids = h_ids[valid_mask]
    
    field_avg = np.mean(valid_rs)
    n_field = len(valid_rs)
    
    exps = 1.0 / (1.0 + 10.0 ** ((field_avg - valid_rs) / 400.0))
    actuals = (n_field - valid_ranks) / (n_field - 1)
    diffs = K * (actuals - exps)
    new_rs = valid_rs + diffs
    
    # Write back to dict
    for h, r_new in zip(valid_hids, new_rs):
        current_ratings[h] = r_new
    
    if race_count < 5:
        print(f"  Updated ratings for {list(valid_hids)[:3]}: {list(new_rs)[:3]}")
    
    race_count += 1

print(f"\n=== Final State ===")
print(f"Total races processed: {race_count}")
print(f"Horses with non-initial ratings: {len(current_ratings)}")
for h in horses:
    final = current_ratings.get(h, INITIAL_RATING)
    print(f"  {h}: {final:.2f}")

# Check target race
target_indices = df_sorted[df_sorted['race_id'] == RACE_ID].index.values
print(f"\nTarget race ({RACE_ID}) ELO values:")
if len(target_indices) > 0:
    for idx in target_indices:
        h = df_sorted.loc[idx, 'horse_id']
        elo = res_buffer[idx]
        print(f"  {h}: {elo:.2f}")

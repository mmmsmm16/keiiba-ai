#!/usr/bin/env python3
"""
Debug nige interaction features for the target race.
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

# Load data
df_raw = loader.load_for_horses(horses, DATE, '2016-01-01', False)

# Feature calculation
pipeline = FeaturePipeline(cache_dir='/workspace/data/features')
# Specifically calculate pace_stats (which creates last_nige_rate) and the feature_pipeline's own logic
df_features = pipeline.load_features(df_raw, ['pace_stats'], force=True)

# Extract today
df_today = df_features[df_features['race_id'] == RACE_ID].copy()

print(f"=== Target Race last_nige_rate Analysis ===")
print(df_today[['horse_number', 'last_nige_rate']].sort_values('last_nige_rate', ascending=False))

n_above_02 = (df_today['last_nige_rate'] > 0.2).sum()
print(f"\nHorses with last_nige_rate > 0.2: {n_above_02}")

if n_above_02 >= 1:
    print("\nCalculation would be:")
    # Simulate feature_pipeline calculation
    df_today['is_candidate'] = (df_today['last_nige_rate'] > 0.2).astype(int)
    race_nige_count_bin = df_today['is_candidate'].sum()
    print(f"race_nige_count_bin: {race_nige_count_bin}")
    
    df_today['race_nige_count_excl'] = race_nige_count_bin - df_today['is_candidate']
    df_today['is_nige_interaction'] = df_today['last_nige_rate'] * df_today['race_nige_count_excl']
    
    print("\nis_nige_interaction results:")
    print(df_today[['horse_number', 'is_candidate', 'race_nige_count_excl', 'is_nige_interaction']].sort_values('is_nige_interaction', ascending=False))
else:
    print("\nNO candidates found today. This explains why today=0.")
    print("If this race really has no nige history horses, today=0 is technically correct for THIS race.")
    print("But hist=(constant) means it's always failing.")

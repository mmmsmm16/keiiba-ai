#!/usr/bin/env python3
"""
List all constant columns and categorize them.
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
df_history = df_features[df_features['race_id'] != RACE_ID].copy()

# Find constant columns
numeric_df = df_today.select_dtypes(include=[np.number])
constant_cols = [c for c in numeric_df.columns if numeric_df[c].nunique() <= 1]

# Expected to be constant (race-level features)
EXPECTED_CONSTANT = [
    'distance', 'odds_10min', 'race_nige_count_bin', 'race_nige_pressure_sum', 
    'front_runner_count', 'odds_ratio_10min', 'odds_ratio_60_10', 'rank_diff_10min', 
    'odds_log_ratio_10min', 'odds_final', 'field_elo_mean', 'struct_early_speed_sum',
    'field_size', 'race_impost_std', 'is_handicap_race_guess', 'weather_code', 'going_code',
    'struct_early_speed_mean', 'pace_expectation_proxy', 'style_entropy', 'odds_60min'
]

# JIT limited
JIT_LIMITED = [
    'jockey_win_rate_30d', 'jockey_top3_rate_30d', 'jockey_win_rate_60d', 'jockey_top3_rate_60d',
    'trainer_win_rate_30d', 'trainer_top3_rate_30d', 'trainer_win_rate_60d', 'trainer_top3_rate_60d',
    'relative_jockey_win_rate_30d_z', 'relative_jockey_win_rate_30d_pct',
    'relative_trainer_win_rate_30d_z', 'relative_trainer_win_rate_30d_pct'
]

print(f"\n=== Constant Columns Analysis ===")
print(f"Total constant: {len(constant_cols)}")

expected = [c for c in constant_cols if c in EXPECTED_CONSTANT]
jit_limited = [c for c in constant_cols if c in JIT_LIMITED]
problem = [c for c in constant_cols if c not in EXPECTED_CONSTANT and c not in JIT_LIMITED]

print(f"Expected constant: {len(expected)}")
print(f"JIT limited: {len(jit_limited)}")
print(f"Problem: {len(problem)}")

print(f"\n=== Problem Columns ({len(problem)}) ===")
for c in sorted(problem):
    today_val = numeric_df[c].iloc[0] if not numeric_df[c].isna().all() else 'NaN'
    # Check history
    if c in df_history.columns:
        hist_nunique = df_history[c].nunique()
        hist_range = f"({df_history[c].min():.2f}~{df_history[c].max():.2f})" if hist_nunique > 1 else "(constant)"
    else:
        hist_range = "(N/A)"
    print(f"  {c}: today={today_val}, hist={hist_range}")

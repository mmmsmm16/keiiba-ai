#!/usr/bin/env python3
"""
List the 'Investigate' columns that are constant.
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
df_features = pipeline.load_features(df_raw, list(pipeline.registry.keys()), force=True)

# Extract today
df_today = df_features[df_features['race_id'] == RACE_ID].copy()

# Find constant columns
numeric_df = df_today.select_dtypes(include=[np.number])
constant_cols = [c for c in numeric_df.columns if numeric_df[c].nunique() <= 1]

EXPECTED = ['distance', 'weather_code', 'going_code', 'track_variant', 'field_size', 'race_impost_std', 'is_handicap_race_guess', 'field_elo_mean', 'struct_early_speed_sum', 'struct_early_speed_mean', 'pace_expectation_proxy', 'style_entropy', 'odds_60min', 'odds_10min', 'race_nige_count_bin', 'race_nige_pressure_sum', 'front_runner_count', 'odds_ratio_10min', 'odds_ratio_60_10', 'rank_diff_10min', 'odds_log_ratio_10min', 'odds_final', 'field_size']
LAG_COLS = [c for c in constant_cols if "lag" in c or "first_" in c or "change" in c or "_365d" in c]
RELATIVE = [c for c in constant_cols if "relative_" in c]
JIT_LIMITED = ['jockey_win_rate_30d', 'jockey_top3_rate_30d', 'jockey_win_rate_60d', 'jockey_top3_rate_60d', 'trainer_win_rate_30d', 'trainer_top3_rate_30d', 'trainer_win_rate_60d', 'trainer_top3_rate_60d']

exclude = set(EXPECTED + LAG_COLS + RELATIVE + JIT_LIMITED)
problem = [c for c in constant_cols if c not in exclude]

print(f"\n=== Problem Columns ({len(problem)}) ===")
for c in sorted(problem):
    val = df_today[c].iloc[0] if not df_today[c].isna().all() else 'NaN'
    print(f"  {c}: {val}")

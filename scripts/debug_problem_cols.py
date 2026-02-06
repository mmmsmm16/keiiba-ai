#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline

RACE_ID = "202606010910"
DATE = "2026-01-25"

def main():
    loader = JraVanDataLoader()
    
    # Get horses for the target race
    r_nen, r_venue, r_kai, r_nichi, r_no = RACE_ID[:4], RACE_ID[4:6], RACE_ID[6:8], RACE_ID[8:10], RACE_ID[10:12]
    q = f"SELECT DISTINCT ketto_toroku_bango FROM jvd_se WHERE kaisai_nen='{r_nen}' AND keibajo_code='{r_venue}' AND kaisai_kai='{r_kai}' AND kaisai_nichime='{r_nichi}' AND race_bango='{r_no}'"
    horses = pd.read_sql(q, loader.engine)['ketto_toroku_bango'].astype(str).str.strip().tolist()

    print(f"Loading data for {len(horses)} horses...")
    df_raw = loader.load_for_horses(horses, DATE, "2016-01-01")
    
    pipeline = FeaturePipeline(cache_dir=os.path.join(project_root, "data", "features"))
    df_features = pipeline.load_features(df_raw, list(pipeline.registry.keys()), force=True)
    
    df_today = df_features[df_features['race_id'] == RACE_ID].copy()
    df_history = df_features[df_features['race_id'] != RACE_ID].copy()
    
    numeric_df = df_today.select_dtypes(include=[np.number])
    constant_cols = [c for c in numeric_df.columns if numeric_df[c].nunique() <= 1]
    
    EXPECTED = ['distance', 'weather_code', 'going_code', 'track_variant', 'field_size', 'field_elo_mean']
    LAG_COLS = [c for c in constant_cols if "lag" in c or "first_" in c or "change" in c or "_365d" in c]
    RELATIVE = [c for c in constant_cols if "relative_" in c]
    
    problem = [c for c in constant_cols if c not in EXPECTED + LAG_COLS + RELATIVE]
    
    print(f"\n=== Problem Columns ({len(problem)}) ===")
    results = []
    for c in sorted(problem):
        today_val = df_today[c].iloc[0] if len(df_today) > 0 else "N/A"
        hist_unique = df_history[c].nunique()
        hist_range = f"({df_history[c].min()}~{df_history[c].max()})" if hist_unique > 1 else "Constant"
        print(f"  {c:30}: today={today_val}, hist={hist_range}")

if __name__ == "__main__":
    main()

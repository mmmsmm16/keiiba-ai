#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline

RACE_ID = "202606010910"
DATE = "2026-01-25"

def main():
    loader = JraVanDataLoader()
    
    # Get horses
    r_nen, r_venue, r_kai, r_nichi, r_no = RACE_ID[:4], RACE_ID[4:6], RACE_ID[6:8], RACE_ID[8:10], RACE_ID[10:12]
    q = f"SELECT DISTINCT ketto_toroku_bango FROM jvd_se WHERE kaisai_nen='{r_nen}' AND keibajo_code='{r_venue}' AND kaisai_kai='{r_kai}' AND kaisai_nichime='{r_nichi}' AND race_bango='{r_no}'"
    horses = pd.read_sql(q, loader.engine)['ketto_toroku_bango'].astype(str).str.strip().tolist()

    df_raw = loader.load_for_horses(horses, DATE, "2016-01-01")
    
    # Simulate Pipeline Merge
    key_cols = ['race_id', 'horse_number', 'horse_id']
    current_df = df_raw[key_cols].copy()
    
    from src.preprocessing.features import rating_elo
    print("Computing rating_elo block...")
    block_df = rating_elo.compute(df_raw)
    
    print(f"Current DF length: {len(current_df)}")
    print(f"Block DF length: {len(block_df)}")
    
    # Check key types
    print("\nKey Types (Current):")
    for k in key_cols: print(f"  {k}: {current_df[k].dtype}")
    print("Key Types (Block):")
    for k in key_cols: print(f"  {k}: {block_df[k].dtype}")
    
    # Perform merge
    cols_to_use = ['horse_elo']
    merged = pd.merge(current_df, block_df[key_cols + cols_to_use], on=key_cols, how='left')
    
    print(f"\nMerged Result (Today's Race):")
    today_merged = merged[merged['race_id'] == RACE_ID]
    print(today_merged[['horse_number', 'horse_elo']].sort_values('horse_number').head(10))
    print(f"Unique Elo in merged today: {today_merged['horse_elo'].unique()}")
    print(f"NaN count in merged today: {today_merged['horse_elo'].isna().sum()}")

if __name__ == "__main__":
    main()

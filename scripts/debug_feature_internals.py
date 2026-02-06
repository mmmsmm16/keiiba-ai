#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.features import rating_elo, pace_features

RACE_ID = "202606010910"
DATE = "2026-01-25"

def main():
    loader = JraVanDataLoader()
    
    # Get horses
    r_nen, r_venue, r_kai, r_nichi, r_no = RACE_ID[:4], RACE_ID[4:6], RACE_ID[6:8], RACE_ID[8:10], RACE_ID[10:12]
    q = f"SELECT DISTINCT ketto_toroku_bango FROM jvd_se WHERE kaisai_nen='{r_nen}' AND keibajo_code='{r_venue}' AND kaisai_kai='{r_kai}' AND kaisai_nichime='{r_nichi}' AND race_bango='{r_no}'"
    horses = pd.read_sql(q, loader.engine)['ketto_toroku_bango'].astype(str).str.strip().tolist()

    print(f"Loading data...")
    df_raw = loader.load_for_horses(horses, DATE, "2016-01-01")
    
    print("\n--- Testing Elo Calculation ---")
    elo_df = rating_elo.compute(df_raw)
    today_elo = elo_df[elo_df['race_id'] == RACE_ID]
    print(f"Target Race Elo Unique Values: {today_elo['horse_elo'].unique()}")
    print(f"Top 5 Elo values in target race:\n{today_elo[['horse_number', 'horse_elo']].sort_values('horse_number').head(5)}")
    
    print("\n--- Testing Pace Features ---")
    # Note: notebook used pace_stats, but registry had pace_features at line 100.
    # Let's check if pace_stats block exists in FeaturePipeline.
    from src.preprocessing.feature_pipeline import FeaturePipeline
    pipeline = FeaturePipeline()
    if "pace_stats" in pipeline.registry:
        pace_df = pipeline.registry["pace_stats"](df_raw)
        today_pace = pace_df[pace_df['race_id'] == RACE_ID]
        print(f"Target Race avg_pci Unique Values: {today_pace['avg_pci'].unique()}")
        print(f"Top 5 avg_pci values in target race:\n{today_pace[['horse_number', 'avg_pci']].sort_values('horse_number').head(5)}")
    else:
        print("pace_stats not found in registry. Checking pace_features...")
        pace_df = pace_features.compute(df_raw)
        # Check if avg_pci is even in pace_features
        if 'avg_pci' in pace_df.columns:
            today_pace = pace_df[pace_df['race_id'] == RACE_ID]
            print(f"Target Race avg_pci Unique Values: {today_pace['avg_pci'].unique()}")
        else:
            print(f"avg_pci not in pace_features. Columns: {pace_df.columns.tolist()}")

if __name__ == "__main__":
    main()

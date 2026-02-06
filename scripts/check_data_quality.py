#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.loader import JraVanDataLoader

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
    
    print(f"Total rows: {len(df_raw)}")
    print(f"Today rows: {len(df_raw[df_raw['race_id'] == RACE_ID])}")
    print(f"History rows: {len(df_raw[df_raw['race_id'] != RACE_ID])}")
    
    # Check key columns for PCI/Elo
    cols = ['horse_id', 'raw_time', 'last_3f', 'rank_str']
    print("\n=== Data Sample (History) ===")
    print(df_raw[df_raw['race_id'] != RACE_ID][cols].head(10))
    
    print("\n=== Missing Values (History) ===")
    print(df_raw[df_raw['race_id'] != RACE_ID][cols].isna().sum())
    
    # Check horse_id format
    first_hist_id = df_raw[df_raw['race_id'] != RACE_ID]['horse_id'].iloc[0]
    print(f"\nExample History horse_id: '{first_hist_id}' (Length: {len(first_hist_id)})")
    
    today_ids = df_raw[df_raw['race_id'] == RACE_ID]['horse_id'].unique()
    print(f"Today horse_ids: {today_ids[:3]}")
    
    match_count = df_raw[df_raw['race_id'] != RACE_ID]['horse_id'].isin(today_ids).sum()
    print(f"History rows matching TODAY's horses: {match_count}")

if __name__ == "__main__":
    main()

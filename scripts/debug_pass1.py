#!/usr/bin/env python3
"""
Debug pass_1 and is_nige calculation.
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
df_history = df_raw[df_raw['race_id'] != RACE_ID]

print(f"=== Running Style Analysis ===")
print(f"History rows: {len(df_history)}")

def is_nige_parse(s):
    if pd.isna(s) or s == '' or s == '00' or s == 0:
        return 0
    try:
        parts = str(s).split('-')
        if len(parts) > 0:
            first_corner = float(parts[0])
            return 1 if first_corner == 1 else 0
    except:
        pass
    return 0

# Check running_style from DB
if 'running_style' in df_history.columns:
    print(f"\nrunning_style (from DB) distribution:")
    rs_counts = df_history['running_style'].value_counts()
    print(rs_counts)
    # Map 1:Nige, 2:Senko, 3:Sashi, 4:Oikomi
else:
    print("\nrunning_style column NOT found in df_history")

# Merge inferred and actual
df_history = df_history.copy()
df_history['is_nige_inferred'] = df_history['pass_1'].apply(is_nige_parse)

print(f"\nComparing Nige detection:")
if 'running_style' in df_history.columns:
    has_either = df_history[(df_history['is_nige_inferred'] == 1) | (df_history['running_style'] == 1)]
    print(f"Total rows with either inferred or actual nige: {len(has_either)}")
    overlap = df_history[(df_history['is_nige_inferred'] == 1) & (df_history['running_style'] == 1)]
    print(f"Overlap (Both detected): {len(overlap)}")
    
    only_inferred = df_history[(df_history['is_nige_inferred'] == 1) & (df_history['running_style'] != 1)]
    print(f"Only inferred as Nige: {len(only_inferred)}")
    
    only_db = df_history[(df_history['is_nige_inferred'] != 1) & (df_history['running_style'] == 1)]
    print(f"Only DB says Nige: {len(only_db)}")
    
    if len(only_db) > 0:
        print("\nSample only-DB Nige (pass_1 values):")
        print(only_db['pass_1'].value_counts().head())
else:
    print(f"Total rows inferred as Nige: {df_history['is_nige_inferred'].sum()}")

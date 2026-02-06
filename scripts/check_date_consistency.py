#!/usr/bin/env python3
import sys
import os
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.loader import JraVanDataLoader

RACE_ID = "202606010910"
DATE = "2026-01-25"

def main():
    loader = JraVanDataLoader()
    df = loader.load_for_horses(['2018104775'], DATE, "2016-01-01")
    
    print(f"Date values (unique): {df['date'].unique()}")
    today_rows = df[df['race_id'] == RACE_ID]
    print(f"Today's date in DF: {today_rows['date'].unique()}")
    
    # Check types
    print(f"Date column type: {df['date'].dtype}")
    
    # Check sort order
    df_sorted = df.sort_values(['date', 'race_id'])
    print("\n--- Last 5 rows of sorted DF ---")
    print(df_sorted[['date', 'race_id', 'horse_id']].tail(5))

if __name__ == "__main__":
    main()

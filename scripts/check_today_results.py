#!/usr/bin/env python3
import sys
import os
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.loader import JraVanDataLoader

RACE_ID = "202606010910"

def main():
    loader = JraVanDataLoader()
    r_nen, r_venue, r_kai, r_nichi, r_no = RACE_ID[:4], RACE_ID[4:6], RACE_ID[6:8], RACE_ID[8:10], RACE_ID[10:12]
    q = f"SELECT corner_1, corner_4, soha_time, kohan_3f, mining_kubun, chokyoshi_code, kishu_code FROM jvd_se WHERE kaisai_nen='{r_nen}' AND keibajo_code='{r_venue}' AND kaisai_kai='{r_kai}' AND kaisai_nichime='{r_nichi}' AND race_bango='{r_no}'"
    df = pd.read_sql(q, loader.engine)
    print(f"Results for {RACE_ID}:")
    print(df.head(10))
    print(f"Total rows: {len(df)}")
    if len(df) > 0:
        print(f"corner_1 unique: {df['corner_1'].unique()}")
        print(f"mining_kubun unique: {df['mining_kubun'].unique()}")
        print(f"trainer_id (chokyoshi_code) unique: {df['chokyoshi_code'].unique()}")

if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np
import logging
from sqlalchemy import create_engine, text
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = create_engine('postgresql://postgres:postgres@host.docker.internal:5433/pckeiba')

def test_fallback():
    date_str = "20260201"
    year_str = f"'{date_str[:4]}'"
    
    # 1. Fetch from jvd_o1
    q_o1 = f"SELECT kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, odds_tansho FROM jvd_o1 WHERE kaisai_nen = {year_str} AND kaisai_tsukihi = '{date_str[4:]}'"
    df_o1_raw = pd.read_sql(q_o1, engine)
    print(f"Loaded {len(df_o1_raw)} records from jvd_o1.")
    
    if df_o1_raw.empty:
        return

    def build_rid(row):
        try:
            return f"{int(float(row['kaisai_nen']))}{int(float(row['keibajo_code'])):02}{int(float(row['kaisai_kai'])):02}{int(float(row['kaisai_nichime'])):02}{int(float(row['race_bango'])):02}"
        except: return None
    
    df_o1_raw['race_id'] = df_o1_raw.apply(build_rid, axis=1)
    
    parsed = []
    for _, row in df_o1_raw.iterrows():
        s, rid = row['odds_tansho'], row['race_id']
        if not isinstance(s, str): continue
        for i in range(0, len(s), 8):
            chunk = s[i:i+8]
            if len(chunk) < 8: break
            try:
                parsed.append({
                    'race_id': rid, 'horse_number': int(chunk[0:2]),
                    'odds_10min': int(chunk[2:6]) / 10.0,
                    'popularity_10min': int(chunk[6:8])
                })
            except: continue
            
    if not parsed:
        print("No odds parsed.")
        return
        
    df_o1_parsed = pd.DataFrame(parsed)
    print(f"Parsed {len(df_o1_parsed)} horse-odds records.")
    print("Sample parsed records for Tokyo 12R (202605010212):")
    print(df_o1_parsed[df_o1_parsed['race_id'] == '202605010212'].head())

    # Simulate df_today
    df_today = pd.DataFrame({
        'race_id': ['202605010212'] * 16,
        'horse_number': range(1, 17),
        'odds_10min': [np.nan] * 16
    })
    
    invalid_odds_mask = df_today['odds_10min'].isna() | (df_today['odds_10min'] <= 0)
    
    df_o1_parsed = df_o1_parsed.rename(columns={'odds_10min': 'odds_10min_o1'})
    df_today = pd.merge(df_today, df_o1_parsed, on=['race_id', 'horse_number'], how='left')
    
    print("\nBefore loc assignment (head 5):")
    print(df_today[['race_id', 'horse_number', 'odds_10min', 'odds_10min_o1']].head())
    
    df_today.loc[invalid_odds_mask, 'odds_10min'] = df_today.loc[invalid_odds_mask, 'odds_10min_o1']
    
    print("\nAfter loc assignment (head 5):")
    print(df_today[['race_id', 'horse_number', 'odds_10min']].head())

if __name__ == "__main__":
    test_fallback()

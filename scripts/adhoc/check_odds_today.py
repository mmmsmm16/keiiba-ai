import sys
sys.path.insert(0, '/workspace')

from src.preprocessing.loader import JraVanDataLoader
import pandas as pd

loader = JraVanDataLoader()

target_year = '2026'
target_mmdd = '0104'

# Simulate the query from production_run_t2.py
q = f"SELECT kaisai_nen, kaisai_tsukihi, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, odds_tansho FROM jvd_o1 WHERE kaisai_nen = '{target_year}' AND kaisai_tsukihi = '{target_mmdd}'"
print(f"Query: {q}")
df_odds = pd.read_sql(q, loader.engine)
print(f"\nFetched {len(df_odds)} rows from jvd_o1")
print(df_odds.head())

if not df_odds.empty:
    rows = []
    for _, row in df_odds.iterrows():
        race_id = row['kaisai_nen'] + row['keibajo_code'] + row['kaisai_kai'] + row['kaisai_nichime'] + row['race_bango']
        odds_str = str(row['odds_tansho'])
        print(f"\nRace: {race_id}, odds_str length: {len(odds_str)}")
        print(f"First 20 chars: {odds_str[:20]}...")
        
        for i in range(0, min(len(odds_str), 20), 4):
            horse_num = i // 4 + 1
            odds_raw = odds_str[i:i+4]
            if odds_raw.isdigit():
                odds_val = int(odds_raw) / 10.0
                print(f"  Horse {horse_num}: {odds_val}")
                rows.append({'race_id': race_id, 'horse_number': horse_num, 'odds_live': odds_val})
    
    print(f"\nTotal parsed rows: {len(rows)}")

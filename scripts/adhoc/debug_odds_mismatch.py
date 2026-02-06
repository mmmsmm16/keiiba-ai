
import pandas as pd
import psycopg2
import sys
import os

# Add workspace to path
sys.path.append('/workspace')
from src.preprocessing.loader import JraVanDataLoader

def main():
    target_date = '20260118'
    
    # 1. Load jvd_ra (Race Info) via Loader
    print(f"Loading Race Info for {target_date}...")
    loader = JraVanDataLoader()
    # Mocking loading for specific date requires raw load or filter
    # Usually loader loads range. Let's load 2026.
    df_raw = loader.load(history_start_date='2026-01-01', skip_odds=True)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    df_day = df_raw[df_raw['date'] == pd.to_datetime(target_date)].copy()
    
    if df_day.empty:
        print("Loader found NO races for this date.")
        return
        
    print(f"Loader found {len(df_day)} rows (horses).")
    unique_races_loader = df_day['race_id'].unique()
    print(f"Unique Races (Loader): {len(unique_races_loader)}")
    print("Sample Race IDs (Loader):", unique_races_loader[:3])

    # 2. Load jvd_o1 (Odds) via Direct Query
    print("\nLoading Odds from jvd_o1...")
    conn_str = "host='host.docker.internal' port=5433 dbname='pckeiba' user='postgres' password='postgres'"
    try:
        conn = psycopg2.connect(conn_str)
        q = f"SELECT kaisai_nen, kaisai_tsukihi, keibajo_code, kaisai_kai, kaisai_nichime, race_bango FROM jvd_o1 WHERE kaisai_nen = '2026' AND kaisai_tsukihi = '0118'"
        df_odds = pd.read_sql(q, conn)
        conn.close()
        
        print(f"Direct Query found {len(df_odds)} races.")
        
        # Construct Race IDs
        # Format: YYYY + PlaceCode(2) + Kai(2) + Nichi(2) + RaceNo(2)
        # Note: Loader creates 16-digit ID.
        # Check if DB columns correspond to this.
        
        generated_ids = []
        for _, row in df_odds.iterrows():
             # Ensure padding
             # Usually columns are strings in pckeiba, but let's be safe
             rid = str(row['kaisai_nen']) + \
                   str(row['keibajo_code']).zfill(2) + \
                   str(row['kaisai_kai']).zfill(2) + \
                   str(row['kaisai_nichime']).zfill(2) + \
                   str(row['race_bango']).zfill(2)
             generated_ids.append(rid)
             
        print("Sample Race IDs (Direct):", generated_ids[:3])
        
        # Compare
        set_loader = set(unique_races_loader)
        set_direct = set(generated_ids)
        
        common = set_loader.intersection(set_direct)
        print(f"\nIntersection: {len(common)}")
        
        if len(common) == 0:
            print("!!! NO MATCHING IDs !!!")
            print("Loader ID Sample: ", unique_races_loader[0])
            print("Direct ID Sample: ", generated_ids[0])
            print("Check Place Codes or Padding.")
            
    except Exception as e:
        print(f"DB Error: {e}")

if __name__ == "__main__":
    main()

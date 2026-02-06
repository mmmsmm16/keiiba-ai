
import os
import sys
import pandas as pd
from sqlalchemy import text

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.preprocessing.loader import JraVanDataLoader

def inspect_db_odds():
    print("Connecting to DB...")
    loader = JraVanDataLoader()
    
    # Target races that had 0 odds in previous checks
    # e.g. 202448210108 (Blue Gold, 2024-01-01)
    target_race_id = '202448210108' 
    
    query = text(f"""
    SELECT 
        r.kaisai_nen || r.keibajo_code || r.kaisai_kai || r.kaisai_nichime || r.race_bango as race_id,
        res.umaban,
        res.bamei,
        res.tansho_odds,
        res.kakutei_chakujun
    FROM jvd_se res
    JOIN jvd_ra r
        ON r.kaisai_nen = res.kaisai_nen
        AND r.keibajo_code = res.keibajo_code
        AND r.kaisai_kai = res.kaisai_kai
        AND r.kaisai_nichime = res.kaisai_nichime
        AND r.race_bango = res.race_bango
    WHERE r.kaisai_nen = '2024'
      AND r.keibajo_code = '48' 
      AND r.kaisai_kai = '02' 
      AND r.kaisai_nichime = '01'
      AND r.race_bango = '08'
    """)
    
    print(f"\nExecuting Query for {target_race_id}...")
    try:
        with loader.engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()
            
        print(f"Found {len(rows)} rows.")
        if len(rows) > 0:
            print(f"{'RaceID':<15} {'Uma':<4} {'Name':<20} {'Odds(Raw)':<10} {'Rank'}")
            print("-" * 60)
            for row in rows:
                print(f"{row[0]:<15} {row[1]:<4} {row[2]:<20} {row[3]:<10} {row[4]}")
        else:
            print("No rows found for this race in DB.")
            
    except Exception as e:
        print(f"Query failed: {e}")

if __name__ == "__main__":
    inspect_db_odds()

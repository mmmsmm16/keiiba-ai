import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
from src.preprocessing.loader import JraVanDataLoader

def main():
    loader = JraVanDataLoader()
    
    # Check count of matches
    print("Checking SQL Join Count...")
    q = """
    SELECT count(*) 
    FROM jvd_ra r 
    JOIN apd_sokuho_o1 o1 
    ON r.kaisai_nen = o1.kaisai_nen 
    AND r.keibajo_code = o1.keibajo_code
    AND r.kaisai_kai = o1.kaisai_kai
    AND r.kaisai_nichime = o1.kaisai_nichime
    AND r.race_bango = o1.race_bango
    """
    try:
        count = pd.read_sql(q, loader.engine).iloc[0,0]
        print(f"Total Matches: {count}")
    except Exception as e:
        print(f"Count Query Failed: {e}")

    # Inspect Hex values for mismatch
    print("\nInspecting Hex Values for a Sample Race...")
    q_hex = """
    WITH ra AS (SELECT * FROM jvd_ra LIMIT 1),
         o1 AS (
             SELECT * FROM apd_sokuho_o1 
             WHERE kaisai_nen = (SELECT kaisai_nen FROM ra)
             AND keibajo_code = (SELECT keibajo_code FROM ra)
             LIMIT 1
         )
    SELECT 
        r.race_bango as ra_val, 
        encode(r.race_bango::bytea, 'hex') as ra_hex,
        o1.race_bango as o1_val,
        encode(o1.race_bango::bytea, 'hex') as o1_hex,
        r.race_bango = o1.race_bango as matches
    FROM ra, o1
    """
    try:
        df = pd.read_sql(q_hex, loader.engine)
        print(df)
    except Exception as e:
        print(f"Hex Query Failed: {e}")

if __name__ == "__main__":
    main()

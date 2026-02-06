import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
from src.preprocessing.loader import JraVanDataLoader

def check_race_id_formats():
    loader = JraVanDataLoader()
    
    # 1. Load Race Data (Small chunk)
    print("Loading race data (jvd_ra)...")
    query_ra = "SELECT race_bango FROM jvd_ra LIMIT 10"
    df_ra = pd.read_sql(query_ra, loader.engine)
    print("Race Bango (jvd_ra):", df_ra['race_bango'].tolist())
    
    # 2. Load Odds Data (apd_sokuho_o1)
    print("Loading odds data (apd_sokuho_o1)...")
    query_o1 = "SELECT race_bango FROM apd_sokuho_o1 LIMIT 10"
    df_o1 = pd.read_sql(query_o1, loader.engine)
    print("Race Bango (apd_sokuho_o1):", df_o1['race_bango'].tolist())
    
    # 3. Check types
    print("\nTypes:")
    print("jvd_ra:", df_ra['race_bango'].dtype)
    print("apd_sokuho_o1:", df_o1['race_bango'].dtype)
    
    # 4. Check Race ID construction in loader.py vs reality
    # Loader uses: CONCAT(r.kaisai_nen, r.keibajo_code, r.kaisai_kai, r.kaisai_nichime, r.race_bango)
    # Odds uses: CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango)
    # If race_bango differs (e.g. '01' vs '1'), CONCAT will produce different IDs.
    
    # Let's try to find a matching record to see actual values
    print("\nChecking a specific race ID match...")
    query_match = """
    WITH ra AS (
        SELECT kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango 
        FROM jvd_ra LIMIT 1
    )
    SELECT 
        ra.race_bango as ra_bango, 
        o1.race_bango as o1_bango,
        ra.race_bango = o1.race_bango as is_equal
    FROM ra
    JOIN apd_sokuho_o1 o1 
    ON ra.kaisai_nen = o1.kaisai_nen 
    AND ra.keibajo_code = o1.keibajo_code 
    AND ra.kaisai_kai = o1.kaisai_kai 
    AND ra.kaisai_nichime = o1.kaisai_nichime
    LIMIT 5
    """
    try:
        df_match = pd.read_sql(query_match, loader.engine)
        if not df_match.empty:
            print("Matched rows:\n", df_match)
        else:
            print("No join matches found on Keys (Nen/Code/Kai/Nichime)!")
    except Exception as e:
        print("Query failed:", e)

if __name__ == "__main__":
    check_race_id_formats()

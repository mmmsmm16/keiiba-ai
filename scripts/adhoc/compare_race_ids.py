import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
from src.preprocessing.loader import JraVanDataLoader

def main():
    loader = JraVanDataLoader()
    
    print("Fetching Race Data (SQL CONCAT ID)...")
    # Fetch components and SQL ID
    q_ra = """
    SELECT 
        kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango,
        CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) as race_id_sql
    FROM jvd_ra LIMIT 5
    """
    df_ra = pd.read_sql(q_ra, loader.engine)
    print("JVD_RA Sample:")
    print(df_ra)
    
    print("\nFetching Odds Data (Components)...")
    q_o1 = """
    SELECT 
        kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango
    FROM apd_sokuho_o1 LIMIT 5
    """
    df_o1 = pd.read_sql(q_o1, loader.engine)
    
    # Simulate Loader Logic (Current)
    # Note: Loader uses simple + without strip in my failed fix (Step 8940)
    # I will verify what happens with/without strip
    
    print("\nSimulating Python Construction (Raw):")
    # Cast to str
    for c in df_o1.columns:
        df_o1[c] = df_o1[c].astype(str)
        
    df_o1['race_id_py'] = (
        df_o1['kaisai_nen'] + 
        df_o1['keibajo_code'] + 
        df_o1['kaisai_kai'] + 
        df_o1['kaisai_nichime'] + 
        df_o1['race_bango']
    )
    
    print("APD_O1 Sample (Python ID):")
    print(df_o1[['race_id_py', 'race_bango']])
    
    print("\nComparing Format (Repr):")
    ra_id = df_ra['race_id_sql'].iloc[0]
    o1_id = df_o1['race_id_py'].iloc[0]
    print(f"RA SQL ID: {repr(ra_id)}")
    print(f"O1 PY ID : {repr(o1_id)}")
    print(f"Length: RA={len(ra_id)}, O1={len(o1_id)}")
    
    # Check for whitespace
    print("O1 components:")
    for c in df_o1.columns[:-1]: # skip race_id_py
        val = df_o1[c].iloc[0]
        print(f"  {c}: {repr(val)}")

if __name__ == "__main__":
    main()

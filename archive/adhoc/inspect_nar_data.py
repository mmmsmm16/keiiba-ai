
import sys
import os
import pandas as pd
from sqlalchemy import text

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.inference.loader import InferenceDataLoader

def main():
    loader = InferenceDataLoader()
    
    # 1. Find a recent NAR race ID
    print("Searching for a recent NAR race (Venue > 10)...")
    # jvd_ra usually PK: kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango
    # Construct ID for display: YYYYMMKKDDDRR (Standard format?)
    # Actually checking schema: usually CONCAT is needed.
    
    query_id = text("""
        SELECT kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango 
        FROM jvd_ra 
        WHERE keibajo_code > '10' 
        ORDER BY kaisai_nen DESC, kaisai_tsukihi DESC 
        LIMIT 1
    """)
    
    with loader.engine.connect() as conn:
        res = conn.execute(query_id).fetchone()
        if not res:
            print("No NAR race found.")
            return
        
        # Unpack
        nen, code, kai, nichi, bango = res
        # Construct race_id (18 digits? or JRA-VAN ID 16?)
        # Standard ID in this project: YYYY + Venue(2) + Kai(2) + Nichi(2) + Race(2)
        # But wait, code is strictly 2 chars?
        race_id = f"{nen}{code}{kai}{nichi}{bango}"
        
        print(f"Found Race: {race_id} (Venue: {code})")
        print(f"Key: {nen}-{code}-{kai}-{nichi}-{bango}")
        
        # WHERE clause for other tables using components (safer than ID if table doesn't have ID col)
        where_clause = f"WHERE kaisai_nen='{nen}' AND keibajo_code='{code}' AND kaisai_kai='{kai}' AND kaisai_nichime='{nichi}' AND race_bango='{bango}'"

        # 2. Check Race Info (RA)
        print("\n[RA] Race Info Table:")
        ra = conn.execute(text(f"SELECT * FROM jvd_ra {where_clause}")).fetchone()
        if ra:
            # Show a few key cols by index or convert to dict if possible
            # Just print existence
            print("  - Exists: Yes")
        else:
            print("  - Exists: NO")

        # 3. Check Result Info (SE)
        print("\n[SE] Race Result Table:")
        se = conn.execute(text(f"SELECT COUNT(*) FROM jvd_se {where_clause}")).fetchone()
        print(f"  - Entry Count: {se[0]}")
        
        # Check if 3F or Corner data exists in SE
        se_row = conn.execute(text(f"SELECT * FROM jvd_se {where_clause} LIMIT 1")).mappings().fetchone()
        if se_row:
             print(f"  - Sample Time: {se_row.get('kakutei_chakujun')}")
             print(f"  - Sample 3F: {se_row.get('halon_time_l3')}")
             print(f"  - Sample Corner: {se_row.get('corner_1')}")
        
        # 4. Check Payout Info (HR)
        print("\n[HR] Payout Table:")
        hr = conn.execute(text(f"SELECT * FROM jvd_hr {where_clause}")).fetchone()
        if hr:
            print("  - Exists: Yes")
        else:
            print("  - Exists: NO (Expected)")

        # 5. Check Odds (O1) - Realtime/Pre-race odds
        print("\n[O1] Odds Table (jvd_o1/apd_sokuho_o1):")
        # jvd_o1 key: kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango
        try:
             o1_check = conn.execute(text(f"SELECT * FROM jvd_o1 {where_clause}")).fetchone()
             if o1_check: print("  - Exists: Yes")
             else: print("  - Exists: NO")
        except:
             print("  - Table jvd_o1 might not support direct query.")

if __name__ == "__main__":
    main()

import sys
import os
import pandas as pd
from sqlalchemy import text
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.preprocessing.loader import JraVanDataLoader

def main():
    loader = JraVanDataLoader()
    target_date = "20260131"
    
    # Check apd_se_jv
    print(f"--- Checking apd_se_jv for {target_date} ---")
    try:
        with loader.engine.connect() as conn:
            # First, see if columns exist like kaisai_nen, etc.
            # Usually apd_* tables might have different schemas.
            q_cols = text("SELECT column_name FROM information_schema.columns WHERE table_name = 'apd_se_jv'")
            cols = [r[0] for r in conn.execute(q_cols).fetchall()]
            print(f"Columns in apd_se_jv: {cols}")
            
            if 'kaisai_nen' in cols and 'kaisai_tsukihi' in cols:
                q = text("SELECT count(*) FROM apd_se_jv WHERE kaisai_nen=:y AND kaisai_tsukihi=:m AND kakutei_chakujun > '00'")
                res = conn.execute(q, {"y": "2026", "m": "0131"}).fetchone()
                print(f"Rows with rank > 0 in apd_se_jv for 1/31: {res[0]}")
    except Exception as e:
        print(f"Error checking apd_se_jv: {e}")

    # Check apd_se_nv
    print(f"\n--- Checking apd_se_nv for {target_date} ---")
    try:
        with loader.engine.connect() as conn:
            q_cols = text("SELECT column_name FROM information_schema.columns WHERE table_name = 'apd_se_nv'")
            cols = [r[0] for r in conn.execute(q_cols).fetchall()]
            print(f"Columns in apd_se_nv: {cols}")
            
            if 'kaisai_nen' in cols and 'kaisai_tsukihi' in cols:
                q = text("SELECT count(*) FROM apd_se_nv WHERE kaisai_nen=:y AND kaisai_tsukihi=:m AND kakutei_chakujun > '00'")
                res = conn.execute(q, {"y": "2026", "m": "0131"}).fetchone()
                print(f"Rows with rank > 0 in apd_se_nv for 1/31: {res[0]}")
    except Exception as e:
        print(f"Error checking apd_se_nv: {e}")

if __name__ == "__main__":
    main()

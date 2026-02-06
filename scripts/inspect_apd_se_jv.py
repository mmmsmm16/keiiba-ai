import sys
import os
import pandas as pd
from sqlalchemy import text
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.preprocessing.loader import JraVanDataLoader

def main():
    loader = JraVanDataLoader()
    target_date = "20260131"
    
    print(f"--- Top 5 rows from apd_se_jv for {target_date} ---")
    try:
        with loader.engine.connect() as conn:
            q = text("SELECT * FROM apd_se_jv WHERE kaisai_nen='2026' AND kaisai_tsukihi='0131' LIMIT 5")
            df = pd.read_sql(q, conn)
            print(df)
            
            # Check if there are ANY rows for 1/31 in this table
            q_count = text("SELECT count(*) FROM apd_se_jv WHERE kaisai_nen='2026' AND kaisai_tsukihi='0131'")
            res = conn.execute(q_count).fetchone()
            print(f"\nTotal rows in apd_se_jv for 1/31: {res[0]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

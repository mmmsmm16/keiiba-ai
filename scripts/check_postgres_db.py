import sys
import os
import pandas as pd
from sqlalchemy import create_engine, text
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

def main():
    # Connection string for 'postgres' DB
    engine = create_engine("postgresql://postgres:postgres@host.docker.internal:5433/postgres")
    
    target_date = "20260131"
    year = target_date[:4]
    mmdd = target_date[4:]
    
    print(f"--- Checking 'postgres' database for {target_date} ---")
    try:
        with engine.connect() as conn:
            # Check jvd_se
            q_se = text(f"SELECT count(*) FROM jvd_se WHERE kaisai_nen=:year AND kaisai_tsukihi=:mmdd AND kakutei_chakujun > '00'")
            res_se = conn.execute(q_se, {"year": year, "mmdd": mmdd}).fetchone()
            print(f"Ranks > 0 in jvd_se: {res_se[0]}")
            
            # Check jvd_hr
            q_hr = text(f"SELECT count(*) FROM jvd_hr WHERE kaisai_nen=:year AND kaisai_tsukihi=:mmdd")
            res_hr = conn.execute(q_hr, {"year": year, "mmdd": mmdd}).fetchone()
            print(f"Payout records in jvd_hr: {res_hr[0]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

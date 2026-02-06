import sys
import os
import pandas as pd
from sqlalchemy import text
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.preprocessing.loader import JraVanDataLoader

def main():
    loader = JraVanDataLoader()
    
    # List all databases
    with loader.engine.connect() as conn:
        q_dbs = text("SELECT datname FROM pg_database WHERE datistemplate = false")
        res_dbs = conn.execute(q_dbs).fetchall()
        print("\nAll databases on this server:")
        print([r[0] for r in res_dbs])

    # List all tables in current DB
    with loader.engine.connect() as conn:
        q_tables = text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name")
        res_tables = conn.execute(q_tables).fetchall()
        print("\nAll tables in public schema:")
        print([r[0] for r in res_tables])

    # Check 1/31 payout
    target_date = "20260131"
    year = target_date[:4]
    mmdd = target_date[4:]
    with loader.engine.connect() as conn:
        q = text(f"SELECT count(*) FROM jvd_hr WHERE kaisai_nen=:year AND kaisai_tsukihi=:mmdd")
        res = conn.execute(q, {"year": year, "mmdd": mmdd}).fetchone()
        print(f"\nNumber of payout records in jvd_hr for {target_date}: {res[0]}")
        
    # Check 1/25 (Sunday)
    target_date_2 = "20260125"
    year_2 = target_date_2[:4]
    mmdd_2 = target_date_2[4:]
    with loader.engine.connect() as conn:
        q = text(f"SELECT count(*) FROM jvd_se WHERE kaisai_nen=:year AND kaisai_tsukihi=:mmdd AND kakutei_chakujun > '00'")
        res = conn.execute(q, {"year": year_2, "mmdd": mmdd_2}).fetchone()
        print(f"Number of horses with results on {target_date_2}: {res[0]}")

if __name__ == "__main__":
    main()

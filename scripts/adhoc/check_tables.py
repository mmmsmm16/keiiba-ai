
import os
import pandas as pd
from sqlalchemy import create_engine, text

import argparse

def check_tables():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", nargs='+', default=['jvd_wf', 'jvd_se'], help="List of tables to check")
    args = parser.parse_args()

    # Database connection
    db_configs = [
        "postgresql://postgres:postgres@localhost:5433/pckeiba",
        "postgresql://postgres:postgres@localhost:5432/pckeiba",
        "postgresql://postgres:postgres@db:5432/pckeiba"
    ]
    
    engine = None
    for conn_str in db_configs:
        try:
            eng = create_engine(conn_str)
            with eng.connect() as conn:
                pass
            engine = eng
            break
        except Exception:
            continue
    
    if not engine:
        print("Could not connect to database.")
        return

    for table in args.tables:
        print(f"\nChecking {table}...")
        try:
            cols = pd.read_sql(text(f"SELECT * FROM {table} LIMIT 0"), engine)
            print(f"{table} columns:", cols.columns.tolist())
        except Exception as e:
            print(f"{table} not found or error: {e}")

if __name__ == "__main__":
    check_tables()

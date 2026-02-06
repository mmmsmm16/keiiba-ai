
import pandas as pd
from sqlalchemy import create_engine
import os
import sys

sys.path.append(os.getcwd())
from src.scripts.auto_predict_v13 import get_db_engine

def main():
    engine = get_db_engine()
    tables = [
        'apd_sokuho_o1', # Win, Place, Bracket?
        'apd_sokuho_o2', # Umaren
        'apd_sokuho_o3', # Wide
        'apd_sokuho_o4', # Umatan
        'apd_sokuho_o5', # Sanrenpuku
        'apd_sokuho_o6', # Sanrentan
    ]
    
    print("| Table | Min Year | Max Year | Count |")
    print("|---|---|---|---|")
    
    for t in tables:
        try:
            # Check if table exists
            # Using raw SQL
            query = f"SELECT MIN(kaisai_nen), MAX(kaisai_nen), COUNT(*) FROM {t}"
            df = pd.read_sql(query, engine)
            print(f"| {t} | {df.iloc[0,0]} | {df.iloc[0,1]} | {df.iloc[0,2]} |")
        except Exception as e:
            print(f"| {t} | Error | Error | {e} |")

if __name__ == "__main__":
    main()

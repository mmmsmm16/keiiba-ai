
import pandas as pd
import sys
import os
from sqlalchemy import create_engine

# Add workspace
sys.path.append('/workspace')
from src.preprocessing.loader import JraVanDataLoader

def main():
    loader = JraVanDataLoader()
    print("Checking DB Date Ranges...")
    
    # Check jvd_ra
    query_ra = "SELECT MAX(kaisai_nengappi) as max_date FROM jvd_ra"
    try:
        df_ra = pd.read_sql(query_ra, loader.engine)
        print(f"jvd_ra Max Date: {df_ra['max_date'].iloc[0]}")
    except Exception as e:
        print(f"jvd_ra Check Failed: {e}")

    # Check apd_sokuho_o1
    query_o1 = "SELECT MAX(kaisai_nen || kaisai_tsukihi) as max_date_str FROM apd_sokuho_o1"
    # Note: kaisai_tsukihi is likely MMDD. kaisai_nen is YYYY.
    try:
        df_o1 = pd.read_sql(query_o1, loader.engine)
        print(f"apd_sokuho_o1 Max Date (YYYYMMDD approx): {df_o1['max_date_str'].iloc[0]}")
    except Exception as e:
        print(f"apd_sokuho_o1 Check Failed: {e}")
        
    # Check 2026 count
    query_count = "SELECT COUNT(*) as cnt FROM apd_sokuho_o1 WHERE kaisai_nen = '2026'"
    try:
        df_count = pd.read_sql(query_count, loader.engine)
        print(f"apd_sokuho_o1 2026 Rows: {df_count['cnt'].iloc[0]}")
    except Exception as e:
        print(f"2026 Count Failed: {e}")

if __name__ == "__main__":
    main()

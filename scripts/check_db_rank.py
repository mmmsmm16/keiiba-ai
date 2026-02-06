import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.preprocessing.loader import JraVanDataLoader

def main():
    loader = JraVanDataLoader()
    target_date = "20260131"
    year = target_date[:4]
    mmdd = target_date[4:]
    
    q = f"""
    SELECT kaisai_nen, kaisai_tsukihi, race_bango, umaban, kakutei_chakujun
    FROM jvd_se 
    WHERE kaisai_nen = '{year}' AND kaisai_tsukihi = '{mmdd}'
    LIMIT 20
    """
    df = pd.read_sql(q, loader.engine)
    print(f"--- Data for {target_date} ---")
    print(df)
    
    # List haraimodoshi columns
    q_cols = "SELECT column_name FROM information_schema.columns WHERE table_name = 'jvd_se' AND column_name LIKE 'haraimodoshi%'"
    df_cols = pd.read_sql(q_cols, loader.engine)
    print("\nHaraimodoshi columns in jvd_se:")
    print(df_cols['column_name'].tolist())
    
    # Also check if any rank is > 0
    q_all = f"SELECT count(*) as count FROM jvd_se WHERE kaisai_nen='{year}' AND kaisai_tsukihi='{mmdd}' AND kakutei_chakujun > 0"
    df_count = pd.read_sql(q_all, loader.engine)
    print(f"\nNumber of entries with rank > 0: {df_count['count'].iloc[0]}")

if __name__ == "__main__":
    main()

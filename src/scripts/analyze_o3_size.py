
import os
import pandas as pd
from sqlalchemy import create_engine, inspect

def main():
    user = os.environ.get('POSTGRES_USER', 'user')
    password = os.environ.get('POSTGRES_PASSWORD', 'password')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(connection_str)
    
    # Check jvd_o3 record count
    print("Checking jvd_o3 record count...")
    query_count = "SELECT COUNT(*) FROM jvd_o3"
    try:
        count = pd.read_sql(query_count, engine).iloc[0,0]
        print(f"Total Rows: {count}")
    except:
        print("Count failed")

    # Check distinct race_ids
    print("\nChecking identifiers...")
    query_ids = """
    SELECT 
        kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango,
        length(odds_umaren) as len_odds,
        toroku_tosu
    FROM jvd_o3 
    LIMIT 5
    """
    df = pd.read_sql(query_ids, engine)
    print(df)

if __name__ == "__main__":
    main()

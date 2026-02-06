import os
import pandas as pd
from sqlalchemy import create_engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_format():
    user = os.environ.get('POSTGRES_USER', 'user')
    password = os.environ.get('POSTGRES_PASSWORD', 'password')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    
    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(connection_str)
    
    # 1. Place (Fukusho) from apd_sokuho_o1
    print("\n=== Place (Fukusho) Sample ===")
    query_place = "SELECT odds_fukusho FROM apd_sokuho_o1 WHERE odds_fukusho IS NOT NULL LIMIT 3"
    try:
        df_place = pd.read_sql(query_place, engine)
        for val in df_place['odds_fukusho']:
            print(f"Raw: {val[:50]}...")
    except Exception as e:
        print(e)

    # 2. Umaren from apd_sokuho_o2
    print("\n=== Umaren Sample ===")
    query_umaren = "SELECT odds_umaren FROM apd_sokuho_o2 WHERE odds_umaren IS NOT NULL LIMIT 3"
    try:
        df_umaren = pd.read_sql(query_umaren, engine)
        for _, row in df_umaren.iterrows():
            print(f"Raw: {row['odds_umaren'][:50]}...")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    inspect_format()

import os
import pandas as pd
from sqlalchemy import create_engine, text

def check_nar_odds():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
    port = os.environ.get('POSTGRES_PORT', '5433')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    
    db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(db_url)
    
    # Check nvd_o1 (Win/Place odds likely)
    # Need to check column names first or just select * limit 5
    # PC-KEIBA usually uses standard column names.
    
    try:
        # First check count
        count_query = text("SELECT count(*) FROM nvd_o1")
        with engine.connect() as conn:
            count = conn.execute(count_query).scalar()
        
        print(f"Total records in nvd_o1: {count}")
        
        if count > 0:
            # Check for data
            # Assuming columns like 'tansho_odds', 'fukusho_odds_low' etc. but let's see raw
            df = pd.read_sql("SELECT * FROM nvd_o1 LIMIT 5", engine)
            print("\nSample Data (nvd_o1):")
            print(df.head())
            
            # Check if fields are all zero?
            # PC-KEIBA NAR odds might be stored in 'odds_tansho' etc.
            # Let's inspect columns from the sample df.
            
    except Exception as e:
        print(f"Error checking nvd_o1: {e}")

    # Also check nvd_ra to see recent dates
    try:
        query_date = text("SELECT MAX(kaisai_nen) as max_year, MAX(kaisai_tsukihi) as max_date FROM nvd_ra")
        with engine.connect() as conn:
            result = conn.execute(query_date).mappings().first()
            print(f"\nLatest NAR Race Date: Year {result['max_year']}, Date {result['max_date']}")
    except Exception as e:
        print(f"Error checking nvd_ra: {e}")

if __name__ == "__main__":
    check_nar_odds()

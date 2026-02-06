
import os
import sys
import pandas as pd
from sqlalchemy import create_engine

def test_sqlalchemy_connection():
    host = os.environ.get("POSTGRES_HOST", "db")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres") 
    db = os.environ.get("POSTGRES_DB", "keiiba")
    port = 5432
    
    db_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    print(f"Connecting to: {db_url.replace(password, '******')}")
    
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            print("SQLAlchemy Connect: SUCCESS")
            
            # Test Query
            df = pd.read_sql("SELECT count(*) FROM jvd_ra LIMIT 1", conn)
            print("pd.read_sql: SUCCESS")
            print(df)
            
    except Exception as e:
        print(f"FAILURE: {e}")

if __name__ == "__main__":
    test_sqlalchemy_connection()


import os
import pandas as pd
from sqlalchemy import create_engine

def list_dbs():
    host = os.environ.get("POSTGRES_HOST", "db")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres") 
    port = 5432
    # Connect to 'postgres' to list dbs
    url = f"postgresql://{user}:{password}@{host}:{port}/postgres"
    
    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            df = pd.read_sql("SELECT datname FROM pg_database WHERE datistemplate = false;", conn)
            print("Databases:")
            print(df)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_dbs()

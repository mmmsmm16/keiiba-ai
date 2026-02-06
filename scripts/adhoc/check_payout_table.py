
import os
import pandas as pd
from sqlalchemy import create_engine, inspect

def check_payout():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
    port = os.environ.get('POSTGRES_PORT', '5433') # Try 5433 as per other scripts, or 5432
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    
    # Try 5432 first (default in loader)
    try:
        connection_str = f"postgresql://{user}:{password}@{host}:5432/{dbname}"
        engine = create_engine(connection_str)
        inspector = inspect(engine)
        tables = inspector.get_table_names(schema='public')
    except:
        # Try 5433
        connection_str = f"postgresql://{user}:{password}@{host}:5433/{dbname}"
        engine = create_engine(connection_str)
        inspector = inspect(engine)
        tables = inspector.get_table_names(schema='public')

    print("Tables found:", len(tables))
    
    target_tables = [t for t in tables if 'haraimodoshi' in t or 'pay' in t or 'jvd_hr' in t]
    print("Potential Payout Tables:", target_tables)
    
    if target_tables:
        t = target_tables[0]
        print(f"Sampling {t}...")
        try:
            df = pd.read_sql(f"SELECT * FROM {t} LIMIT 5", engine)
            print(df.columns.tolist())
            print(df.head())
        except Exception as e:
            print(e)
            
if __name__ == "__main__":
    check_payout()

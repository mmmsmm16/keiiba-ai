
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

    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names(schema='public')
        print("Tables in database:")
        for t in tables:
            if 'jvd_' in t or 'chokyo' in t:
                print(f"- {t}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

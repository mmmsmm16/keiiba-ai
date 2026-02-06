
import os
import pandas as pd
from sqlalchemy import create_engine
import logging

def main():
    user = os.environ.get('POSTGRES_USER', 'user')
    password = os.environ.get('POSTGRES_PASSWORD', 'password')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    
    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(connection_str)

    try:
        # Limit 1 to get column names
        query = "SELECT * FROM jvd_wc LIMIT 1"
        df = pd.read_sql(query, engine)
        print("Columns in jvd_wc:")
        for col in df.columns:
            print(f"- {col}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

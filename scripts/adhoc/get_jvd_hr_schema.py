
import os
import pandas as pd
from sqlalchemy import create_engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'postgres')
    
    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(connection_str)
    
    query = "SELECT * FROM jvd_hr LIMIT 1"
    
    try:
        df = pd.read_sql(query, engine)
        print("Table: jvd_hr")
        print(df.columns.tolist())
        
        # Also print first row to see data format
        print("\nFirst row sample:")
        print(df.iloc[0].to_dict())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

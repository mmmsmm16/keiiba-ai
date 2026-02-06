import os
import pandas as pd
from sqlalchemy import create_engine, inspect
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_tables():
    user = os.environ.get('POSTGRES_USER', 'user')
    password = os.environ.get('POSTGRES_PASSWORD', 'password')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    
    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(connection_str)
    
    inspector = inspect(engine)
    tables = inspector.get_table_names(schema='public')
    
    targets = ['apd_sokuho_o1', 'apd_sokuho_o2', 'apd_sokuho_o3']
    
    for t in targets:
        if t in tables:
            print(f"\n=== Table: {t} ===")
            columns = [c['name'] for c in inspector.get_columns(t)]
            print(f"Columns for {t}:")
            for c in columns:
                print(f"  - {c}")
            
            # Sample data
            # df = pd.read_sql(f"SELECT * FROM {t} LIMIT 1", engine)
            # print(df.T)
        else:
            print(f"Table {t} not found.")

if __name__ == "__main__":
    inspect_tables()


import os
import pandas as pd
from sqlalchemy import create_engine, inspect

def list_db_tables():
    # Use environment provided to the script
    host = os.environ.get("POSTGRES_HOST", "host.docker.internal")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres") 
    db = os.environ.get("POSTGRES_DB", "postgres")
    port = 5432
    
    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    print(f"Connecting to: {user}@{host}:{port}/{db}")
    
    try:
        engine = create_engine(url)
        inspector = inspect(engine)
        
        schemas = inspector.get_schema_names()
        print(f"Schemas: {schemas}")
        
        for schema in schemas:
            if schema in ['information_schema', 'pg_catalog', 'pg_toast']: continue
            tables = inspector.get_table_names(schema=schema)
            print(f"--- Schema: {schema} ---")
            print(tables)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_db_tables()

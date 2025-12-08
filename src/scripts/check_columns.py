
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

    inspector = inspect(engine)
    
    targets = ['jvd_se', 'jvd_ra', 'apd_sokuho_o1', 'apd_sokuho_ra', 'apd_sokuho_se']
    
    for t in targets:
        print(f"\nTABLE: {t}")
        try:
            columns = inspector.get_columns(t)
            col_names = [c['name'] for c in columns]
            print(f"Columns: {col_names}")
            
            # If it's apd_sokuho_o1, let's see some data samples if possible (using pandas)
            # but getting columns is enough for now.
        except Exception as e:
            print(f"Error inspecting {t}: {e}")

if __name__ == "__main__":
    main()

import os
from sqlalchemy import create_engine, inspect

def diagnose():
    user = os.environ.get('POSTGRES_USER', 'user')
    password = os.environ.get('POSTGRES_PASSWORD', 'password')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    
    url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(url)
    inspector = inspect(engine)
    
    tables = inspector.get_table_names()
    print(f"Tables: {tables}")
    
    target_tables = ['jvd_ra', 'jvd_se', 'jvd_um', 'jvd_race_shosai', 'seiseki', 'uma_master']
    
    for table in target_tables:
        if table in tables:
            cols = [c['name'] for c in inspector.get_columns(table)]
            print(f"\n--- Table: {table} ({len(cols)} columns) ---")
            print(f"Columns: {sorted(cols)}")
            
            # Check for candidate columns
            for pattern in ['grade', 'joken', 'shubetsu', 'kubun', 'title', 'mei']:
                matches = [c for c in cols if pattern in c]
                if matches:
                    print(f"  Matches for '{pattern}': {matches}")
        else:
            print(f"\n--- Table: {table} NOT FOUND ---")

if __name__ == "__main__":
    diagnose()

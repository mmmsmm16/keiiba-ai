
import os
import sys
from sqlalchemy import create_engine, inspect

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def list_rt_tables():
    user = os.environ.get('POSTGRES_USER', 'user')
    password = os.environ.get('POSTGRES_PASSWORD', 'password')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    
    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    print(f"Connecting to {dbname}...")
    
    try:
        engine = create_engine(connection_str)
        inspector = inspect(engine)
        tables = inspector.get_table_names(schema='public')
        
        print(f"Total tables found: {len(tables)}")
        
        rt_tables = [t for t in tables if 'rt' in t or 'sokuho' in t or 'odds' in t]
        
        print("\nPossible Real-Time/Odds Tables:")
        for t in sorted(rt_tables):
            print(f"- {t}")
            
            # Inspect columns for interesting tables
            if t.startswith('jvd_rt') or 'tan' in t:
                cols = inspector.get_columns(t)
                col_names = [c['name'] for c in cols]
                print(f"  Columns: {col_names[:10]}...") 

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_rt_tables()

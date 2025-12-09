
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
    
    # Check O2, O4
    print("=== O2/O4 Check ===")
    for t in ['jvd_o2', 'jvd_o4']:
        try:
            cols = [c['name'] for c in inspector.get_columns(t)]
            odds_cols = [c for c in cols if 'odd' in c]
            print(f"{t}: {odds_cols}")
        except:
            print(f"{t}: Error")
            
    # Sample O5 content (Sanrenpuku)
    print("\n=== O5 Content ===")
    try:
        query = "SELECT odds_sanrenpuku FROM jvd_o5 LIMIT 1"
        df = pd.read_sql(query, engine)
        if not df.empty:
            val = df.iloc[0]['odds_sanrenpuku']
            print(f"Len: {len(str(val))}")
            print(f"Head 300: {str(val)[:300]}")
    except:
        print("O5 sample failed")

if __name__ == "__main__":
    main()

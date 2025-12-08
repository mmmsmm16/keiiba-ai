
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
    
    # Check jvd_se
    print("=== jvd_se ===")
    try:
        cols = [c['name'] for c in inspector.get_columns('jvd_se')]
        print(f"Columns: {cols}")
        if 'tansho_odds' in cols:
            print("FOUND: tansho_odds in jvd_se")
    except Exception as e:
        print(e)
        
    # Check apd_sokuho_o1
    print("\n=== apd_sokuho_o1 ===")
    try:
        cols = [c['name'] for c in inspector.get_columns('apd_sokuho_o1')]
        print(f"Columns: {cols}")
        
        # Sample data
        query = "SELECT * FROM apd_sokuho_o1 LIMIT 1"
        df = pd.read_sql(query, engine)
        if not df.empty:
            print("Sample Row:")
            print(df.iloc[0].to_dict())
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()


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
    cols = [c['name'] for c in inspector.get_columns('jvd_o3')]
    print("Columns:", cols)
    
    if 'odds_umaren' in cols:
        query = "SELECT odds_umaren FROM jvd_o3 LIMIT 1"
        df = pd.read_sql(query, engine)
        if not df.empty:
            val = df.iloc[0]['odds_umaren']
            print(f"Type: {type(val)}")
            print(f"Length: {len(str(val))}")
            print(f"Head 200: {str(val)[:200]}")
    else:
        print("odds_umaren column not found")

if __name__ == "__main__":
    main()


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
    cols = inspector.get_columns('jvd_o1')
    for c in cols:
        if c['name'] == 'odds_tansho':
            print(f"Column: odds_tansho, Type: {c['type']}")
            
    # Sample value
    query = "SELECT odds_tansho FROM jvd_o1 LIMIT 1"
    try:
        df = pd.read_sql(query, engine)
        if not df.empty:
            val = df.iloc[0]['odds_tansho']
            print(f"Value: {val}")
            print(f"Length: {len(str(val))}")
        else:
            print("Empty table")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()


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

    try:
        query = "SELECT * FROM apd_sokuho_o1 LIMIT 1"
        df = pd.read_sql(query, engine)
        if df.empty:
            print("apd_sokuho_o1 is empty.")
        else:
            print("apd_sokuho_o1 columns:")
            for c in df.columns:
                print(c)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()

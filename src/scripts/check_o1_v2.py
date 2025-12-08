
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

    # Check jvd_o1
    print("=== jvd_o1 ===")
    try:
        query = "SELECT * FROM jvd_o1 LIMIT 1"
        df = pd.read_sql(query, engine)
        if df.empty:
            print("jvd_o1 is empty.")
        else:
            print("jvd_o1 columns:")
            for c in df.columns:
                print(c)
    except Exception as e:
        print(e)
        
    # Check jvd_se columns again
    print("\n=== jvd_se ===")
    try:
        query = "SELECT * FROM jvd_se LIMIT 1"
        df = pd.read_sql(query, engine)
        if df.empty:
            print("jvd_se is empty.")
        else:
            print("jvd_se columns (sample):")
            # print only interesting columns
            interesting = [c for c in df.columns if 'odd' in c or 'ninki' in c]
            print(interesting)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()

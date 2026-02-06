
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

    # Check 'jvd_se' and 'apd_sokuho_se'
    for tbl in ['jvd_se', 'apd_sokuho_se']:
        print(f"--- {tbl} Date Check ---")
        try:
            query = f"SELECT MAX(CONCAT(kaisai_nen, kaisai_tsukihi)) as max_date FROM {tbl}"
            df = pd.read_sql(query, engine)
            print(df)
        except Exception as e:
            print(f"Error ({tbl}): {e}")


if __name__ == "__main__":
    main()

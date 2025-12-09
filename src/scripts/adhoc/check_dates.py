
import os
import pandas as pd
from sqlalchemy import create_engine

def main():
    user = os.environ.get('POSTGRES_USER', 'user')
    password = os.environ.get('POSTGRES_PASSWORD', 'password')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(connection_str)

    try:
        # Check max date in jvd_o1
        q_o1 = "SELECT MAX(kaisai_nen || kaisai_tsukihi) as max_date FROM jvd_o1"
        try:
            df_o1 = pd.read_sql(q_o1, engine)
            print(f"Max Date in jvd_o1: {df_o1.iloc[0]['max_date']}")
        except:
             print("jvd_o1 max date check failed")

        # Check max date in jvd_se
        q_se = "SELECT MAX(kaisai_nen || kaisai_tsukihi) as max_date FROM jvd_se"
        try:
             df_se = pd.read_sql(q_se, engine)
             print(f"Max Date in jvd_se: {df_se.iloc[0]['max_date']}")
        except:
             print("jvd_se max date check failed")

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()

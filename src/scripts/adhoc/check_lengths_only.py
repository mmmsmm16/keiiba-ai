
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
    
    targets = {
        'jvd_o2': 'odds_umaren',
        'jvd_o3': 'odds_wide',
        'jvd_o4': 'odds_umatan',
        'jvd_o5': 'odds_sanrenpuku',
        'jvd_o6': 'odds_sanrentan'
    }
    
    for t, col in targets.items():
        try:
            query = f"SELECT length({col}) as len FROM {t} LIMIT 1"
            df = pd.read_sql(query, engine)
            if not df.empty:
                print(f"{t}: {df.iloc[0]['len']}")
            else:
                print(f"{t}: Empty")
        except:
             print(f"{t}: Error")

if __name__ == "__main__":
    main()

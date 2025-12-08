
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
        'jvd_o2': {'name': 'Umaren', 'combos': 153, 'col': 'odds_umaren'},
        'jvd_o3': {'name': 'Wide', 'combos': 153, 'col': 'odds_wide'},
        'jvd_o4': {'name': 'Umatan', 'combos': 306, 'col': 'odds_umatan'},
        'jvd_o5': {'name': 'Sanrenpuku', 'combos': 816, 'col': 'odds_sanrenpuku'},
        'jvd_o6': {'name': 'Sanrentan', 'combos': 4896, 'col': 'odds_sanrentan'}
    }
    
    for t, info in targets.items():
        print(f"\n=== {t} ({info['name']}) ===")
        try:
            query = f"SELECT length({info['col']}) as len, {info['col']} as val FROM {t} LIMIT 1"
            df = pd.read_sql(query, engine)
            if not df.empty:
                l = df.iloc[0]['len']
                val = df.iloc[0]['val']
                combos = info['combos']
                print(f"Length: {l}")
                print(f"Est Bytes/Combo: {l / combos}")
                print(f"Sample: {str(val)[:50]}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

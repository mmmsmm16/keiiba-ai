
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
    
    targets = ['jvd_o3', 'jvd_o5', 'jvd_o6']
    
    for t in targets:
        print(f"\n=== {t} ===")
        try:
            # Columns
            query_col = f"SELECT * FROM {t} LIMIT 1"
            df = pd.read_sql(query_col, engine)
            print("Columns:", df.columns.tolist())
            
            # Identify odds column
            # Expected: odds_umaren, odds_sanrenpuku, odds_sanrentan
            odds_col = [c for c in df.columns if 'odd' in c]
            print("Odds Cols:", odds_col)
            
            if odds_col:
                tgt_col = odds_col[0]
                val = df.iloc[0][tgt_col]
                print(f"Sample {tgt_col} type: {type(val)}")
                print(f"Sample len: {len(str(val))}")
                print(f"Sample data (head 50): {str(val)[:50]}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

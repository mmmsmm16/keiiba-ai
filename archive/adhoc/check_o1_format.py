
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

    print("=== jvd_o1 odds columns ===")
    try:
        # Get all columns
        inspector = inspect(engine)
        cols = [c['name'] for c in inspector.get_columns('jvd_o1')]
        
        # Check for wide format
        wide_cols = [c for c in cols if 'tansho_odds_' in c]
        if wide_cols:
            print(f"Wide Format Detected. Sample: {wide_cols[:5]}")
        else:
            # Check for normalized format
            if 'tansho_odds' in cols and 'umaban' in cols:
                print("Normalized Format Detected (umaban, tansho_odds)")
            else:
                print("Unknown Format. Listing first 20 cols:")
                print(cols[:20])
                
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()

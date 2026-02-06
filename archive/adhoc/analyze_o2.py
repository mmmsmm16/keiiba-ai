
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
    
    # Check JVD_O2 (Umaren)
    print("Checking JVD_O2...")
    query = """
    SELECT 
        kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango,
        length(odds_umaren) as len,
        odds_umaren
    FROM jvd_o2
    LIMIT 3
    """
    df = pd.read_sql(query, engine)
    
    for i, row in df.iterrows():
        print(f"Race: {row['kaisai_nen']}-{row['keibajo_code']}-{row['race_bango']}")
        print(f"Length: {row['len']}")
        val = row['odds_umaren']
        print(f"Head: {val[:50]}")
        print(f"Tail: {val[-20:]}")
        
        # Check if length is constant
        # For Umaren (18 horses max), combinations = 153.
        # If length is e.g. 153 * X?
        # Try dividing
        if row['len'] and row['len'] > 0:
            print(f"Div 153: {row['len'] / 153}")
            print(f"Div 306: {row['len'] / 306}") # Umatan count

if __name__ == "__main__":
    main()

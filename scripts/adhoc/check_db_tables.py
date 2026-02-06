import os
import pandas as pd
from sqlalchemy import create_engine, text

def check_tables():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
    port = os.environ.get('POSTGRES_PORT', '5433')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    
    db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(db_url)
    
    query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """
    
    try:
        df = pd.read_sql(query, engine)
        print("Existing Tables:")
        print(df['table_name'].tolist())
        
        # Check specific NAR likelihood
        nar_tables = [t for t in df['table_name'] if 'nar' in t or 'chiho' in t or t.startswith('n_')]
        if nar_tables:
            print(f"\nPotential NAR tables found: {nar_tables}")
        else:
            print("\nNo obvious NAR/Chiho tables found.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_tables()

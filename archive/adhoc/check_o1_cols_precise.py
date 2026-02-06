
import os
from sqlalchemy import create_engine, inspect

def main():
    user = os.environ.get('POSTGRES_USER', 'user')
    password = os.environ.get('POSTGRES_PASSWORD', 'password')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(connection_str)

    inspector = inspect(engine)
    cols = [c['name'] for c in inspector.get_columns('jvd_o1')]
    
    if 'tansho_odds_01' in cols:
        print("CONFIRMED: tansho_odds_01 exists")
    elif 'tansho_odds_1' in cols:
        print("CONFIRMED: tansho_odds_1 exists")
    else:
        print("Pattern not found. Searching...")
        matches = [c for c in cols if 'tansho_odds' in c]
        print(f"Matches: {matches[:5]}")

if __name__ == "__main__":
    main()

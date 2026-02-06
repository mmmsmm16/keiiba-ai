
import os
import sys
from sqlalchemy import create_engine, inspect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def check_odds_columns():
    user = os.environ.get('POSTGRES_USER', 'user')
    password = os.environ.get('POSTGRES_PASSWORD', 'password')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    
    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(connection_str)
    inspector = inspect(engine)
    
    tables = [f'apd_sokuho_o{i}' for i in range(1, 7)]
    
    for t in tables:
        if not inspector.has_table(t):
            print(f"{t}: NOT FOUND")
            continue
            
        cols = inspector.get_columns(t)
        odds_cols = [c['name'] for c in cols if 'odds' in c['name']]
        print(f"{t}: {odds_cols}")

if __name__ == "__main__":
    check_odds_columns()

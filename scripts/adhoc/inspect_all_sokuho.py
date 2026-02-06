
import os
import sys
from sqlalchemy import create_engine, inspect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def inspect_all_sokuho():
    user = os.environ.get('POSTGRES_USER', 'user')
    password = os.environ.get('POSTGRES_PASSWORD', 'password')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    
    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    
    target_tables = [
        'apd_sokuho_o1', # Win/Place (Encapsulated in previous check, but confirming Wakuren is here)
        'apd_sokuho_o2', # Suspected Umaren
        'apd_sokuho_o3', # Suspected Wide
        'apd_sokuho_o4', # Suspected Trio
        'apd_sokuho_o5', # Suspected Trifecta
        'apd_sokuho_o6'  # Exacta?
    ]
    
    try:
        engine = create_engine(connection_str)
        inspector = inspect(engine)
        
        for t in target_tables:
            print(f"\n=== {t} ===")
            if not inspector.has_table(t):
                print("  (Table not found)")
                continue
                
            cols = inspector.get_columns(t)
            col_names = [c['name'] for c in cols]
            print(f"  Cols: {col_names}")
            
            # Identify specific odds columns
            odds_cols = [c for c in col_names if 'odds' in c]
            print(f"  Odds Cols: {odds_cols}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_all_sokuho()

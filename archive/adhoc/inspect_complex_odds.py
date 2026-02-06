
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

    # Tables to check
    targets = ['jvd_o2', 'jvd_o3', 'jvd_o4', 'jvd_o5', 'jvd_o6']
    
    for t in targets:
        print(f"\n=== TABLE: {t} ===")
        try:
            cols = [c['name'] for c in inspector.get_columns(t)]
            # Filter interesting columns to avoid huge dump
            odds_cols = [c for c in cols if 'odd' in c]
            print(f"Odds Columns (Sample): {odds_cols[:5]}")
            
            # Check for wide format (numbered)
            numbered = [c for c in cols if c[-2:].isdigit()]
            if numbered:
                 print(f"Numbered Cols Detect: {numbered[:3]} ...")
            else:
                 print("No numbered columns detected.")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

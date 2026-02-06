
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
    tables = inspector.get_table_names(schema='public')
    
    odds_tables = [t for t in tables if t.startswith('jvd_o') or t.startswith('apd_sokuho_o')]
    print("Found Odds Tables:", sorted(odds_tables))

if __name__ == "__main__":
    main()

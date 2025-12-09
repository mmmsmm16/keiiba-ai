
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

    inspector = inspect(engine)
    tables = inspector.get_table_names(schema='public')
    print("ALL TABLES:")
    for t in tables:
        print(f"- {t}")

    # Check for odds related tables
    odds_tables = [t for t in tables if 'odd' in t.lower() or 'oz' in t.lower()]
    print("\nODDS RELATED TABLES:")
    for t in odds_tables:
        print(f"Table: {t}")
        columns = [c['name'] for c in inspector.get_columns(t)]
        print(f"  Columns: {columns[:10]}...") # Show first 10 columns

    # Check jvd_uma_race columns just in case
    print("\nJVD_UMA_RACE Columns:")
    if 'jvd_uma_race' in tables:
        columns = [c['name'] for c in inspector.get_columns('jvd_uma_race')]
        print(columns)
    elif 'jvd_ur' in tables:
        columns = [c['name'] for c in inspector.get_columns('jvd_ur')]
        print(columns)

if __name__ == "__main__":
    main()

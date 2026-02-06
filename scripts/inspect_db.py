
import os
import pandas as pd
from sqlalchemy import create_engine, inspect

user = os.environ.get('POSTGRES_USER', 'user')
password = os.environ.get('POSTGRES_PASSWORD', 'password')
host = os.environ.get('POSTGRES_HOST', 'db')
port = os.environ.get('POSTGRES_PORT', '5432')
dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
engine = create_engine(connection_str)

inspector = inspect(engine)
tables = inspector.get_table_names(schema='public')
print("Tables:", tables)

target_table = 'nvd_se' if 'nvd_se' in tables else 'nvd_seiseki'
if target_table in tables:
    columns = [c['name'] for c in inspector.get_columns(target_table)]
    print(f"\nColumns in {target_table}:")
    for col in columns:
        if 'odds' in col or 'tansho' in col or 'fukusho' in col:
            print(col)
else:
    print("Seiseki table not found")

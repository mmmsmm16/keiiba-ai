"""Check actual table names in DB"""
from sqlalchemy import create_engine, inspect
import os

user = os.environ.get('POSTGRES_USER', 'postgres')
password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
port = os.environ.get('POSTGRES_PORT', '5433')
dbname = os.environ.get('POSTGRES_DB', 'pckeiba')

engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")
inspector = inspect(engine)

print("=== Tables containing 'sokuho' ===")
for table in inspector.get_table_names():
    if 'sokuho' in table.lower():
        print(f"  {table}")

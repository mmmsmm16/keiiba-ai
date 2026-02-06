from sqlalchemy import create_engine, inspect
import os
import logging

logging.basicConfig(level=logging.INFO)

user = os.environ.get('POSTGRES_USER', 'user')
password = os.environ.get('POSTGRES_PASSWORD', 'password')
host = os.environ.get('POSTGRES_HOST', 'db')
port = os.environ.get('POSTGRES_PORT', '5432')
dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
engine = create_engine(connection_str)
inspector = inspect(engine)

print("=== jvd_se (Seiseki) Columns ===")
for col in inspector.get_columns('jvd_se'):
    print(col['name'], col['type'])

print("\n=== jvd_ra (Race) Columns ===")
for col in inspector.get_columns('jvd_ra'):
    print(col['name'], col['type'])

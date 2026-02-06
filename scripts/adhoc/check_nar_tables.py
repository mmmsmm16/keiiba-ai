
import os
from sqlalchemy import create_engine, text

# Setup DB
user = os.environ.get('POSTGRES_USER', 'postgres')
password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
port = os.environ.get('POSTGRES_PORT', '5433')
dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
engine = create_engine(db_url)

query = text("SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'nvd_%'")
print("--- NAR Tables ---")
with engine.connect() as conn:
    rows = conn.execute(query).fetchall()
    for r in rows:
        print(r[0])

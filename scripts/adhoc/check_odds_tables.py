
import os
import pandas as pd
from sqlalchemy import create_engine, text

# Setup DB
user = os.environ.get('POSTGRES_USER', 'postgres')
password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
port = os.environ.get('POSTGRES_PORT', '5433')
dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
engine = create_engine(db_url)

print("--- NVD_O1 (Win/Place/Wakuren) ---")
try:
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM nvd_o1 LIMIT 1"), conn)
        print(df.T)
except Exception as e:
    print(f"Error reading nvd_o1: {e}")

print("\n--- NVD_O2 (Umaren) ---")
try:
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM nvd_o2 LIMIT 1"), conn)
        print(df.T)
except Exception as e:
    print(f"Error reading nvd_o2: {e}")

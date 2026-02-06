
import pandas as pd
import os
from sqlalchemy import create_engine
import sys

def get_db_engine():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'postgres')
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

# Check T1 Features
path = "data/temp_t1/T1_features_2024_2025.parquet"
if os.path.exists(path):
    df = pd.read_parquet(path)
    # T1 doesn't have 'date' usually, it has 'race_id'. Need to parse.
    # But wait, race_id starts with YYYY.
    df['year'] = df['race_id'].astype(str).str[:4].astype(int)
    print(f"T1 File: {path}")
    print(f"2024 rows: {len(df[df['year'] == 2024])}")
    print(f"2025 rows: {len(df[df['year'] == 2025])}")
else:
    print(f"T1 File not found: {path}")

# Check DB
try:
    engine = get_db_engine()
    query = "SELECT count(*) FROM jvd_ra WHERE kaisai_nen = '2025'"
    count = pd.read_sql(query, engine).iloc[0,0]
    print(f"DB jvd_ra 2025 count: {count}")
    
    query_pay = "SELECT count(*) FROM jvd_hr WHERE kaisai_nen = '2025'"
    count_pay = pd.read_sql(query_pay, engine).iloc[0,0]
    print(f"DB jvd_hr 2025 count: {count_pay}")
except Exception as e:
    print(f"DB Error: {e}")

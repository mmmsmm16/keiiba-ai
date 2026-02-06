"""Debug exacta payout data"""
from sqlalchemy import create_engine
import pandas as pd
import os

user = os.getenv("POSTGRES_USER", "postgres")
pw = os.getenv("POSTGRES_PASSWORD", "postgres")
host = os.getenv("POSTGRES_HOST", "host.docker.internal")
port = os.getenv("POSTGRES_PORT", "5433")
db = os.getenv("POSTGRES_DB", "pckeiba")
e = create_engine(f'postgresql://{user}:{pw}@{host}:{port}/{db}')

df = pd.read_sql("SELECT haraimodoshi_umatan_1a, haraimodoshi_umatan_1b, haraimodoshi_umatan_1c FROM jvd_hr WHERE kaisai_nen = '2023' AND haraimodoshi_umatan_1c > '0' LIMIT 10", e)
print("Exacta payout data sample:")
print(df)
print("\nData types:")
print(df.dtypes)
print("\nValues:")
for _, row in df.iterrows():
    print(f"  1st: {row['haraimodoshi_umatan_1a']}, 2nd: {row['haraimodoshi_umatan_1b']}, Pay: {row['haraimodoshi_umatan_1c']}")

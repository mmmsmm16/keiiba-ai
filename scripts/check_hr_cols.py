"""Check jvd_hr columns"""
from sqlalchemy import create_engine
import pandas as pd
import os

user = os.environ.get('POSTGRES_USER', 'postgres')
password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
port = os.environ.get('POSTGRES_PORT', '5433')
dbname = os.environ.get('POSTGRES_DB', 'pckeiba')

engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")
df = pd.read_sql("SELECT * FROM jvd_hr LIMIT 1", engine)
print("Columns:")
for col in df.columns:
    print(f"  {col}")

# Show sample row
print("\nSample row (first 20 cols):")
print(df.iloc[0, :20].to_string())

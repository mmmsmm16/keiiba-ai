
import os
import sys
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

# Query a sample race from nvd_hr
query = text("""
    SELECT *
    FROM nvd_hr
    LIMIT 1
""")

print("--- NVD_HR Columns that look like odds/payouts ---")
with engine.connect() as conn:
    df = pd.read_sql(query, conn)
    for c in df.columns:
        if 'tansho' in c or 'fukusho' in c:
            print(f"{c}: {df[c].iloc[0]}")

print("\n--- Summary ---")
print("Does it look like it has 'all horses' or just 'winners'?")
print("Usually haraimodoshi_tansho_1a/1b means 1st place horse and payout.")
print("It does NOT list horse 2, 3, 4 etc if they didn't win.")

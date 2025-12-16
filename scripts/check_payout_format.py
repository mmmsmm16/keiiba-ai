"""Check jvd_hr payout format"""
from sqlalchemy import create_engine
import pandas as pd
import os

user = os.environ.get('POSTGRES_USER', 'postgres')
password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
port = os.environ.get('POSTGRES_PORT', '5433')
dbname = os.environ.get('POSTGRES_DB', 'pckeiba')

engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

# Get a 2025 row
df = pd.read_sql("""
SELECT 
    kaisai_nen, keibajo_code, race_bango,
    haraimodoshi_tansho_1a,
    haraimodoshi_tansho_1b,
    haraimodoshi_tansho_1c,
    haraimodoshi_tansho_2a,
    haraimodoshi_tansho_2b,
    haraimodoshi_tansho_2c
FROM jvd_hr
WHERE kaisai_nen = '2025'
LIMIT 5
""", engine)

print("Sample 2025 Win Payouts:")
print(df.to_string())
print()
print("Interpretation:")
print("  1a = 馬番 (horse number)")
print("  1b = 払戻金 (payout per 100 yen)")
print("  1c = 人気 (popularity)")

"""Check umaren/wide payout columns in jvd_hr"""
import pandas as pd
from sqlalchemy import create_engine, text

e = create_engine('postgresql://postgres:postgres@db:5432/pckeiba')

# Get all columns from jvd_hr
cols = pd.read_sql(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'jvd_hr' ORDER BY ordinal_position"), e)
all_cols = cols['column_name'].tolist()

print("=== jvd_hr columns related to payouts ===")
payout_cols = [c for c in all_cols if 'umaren' in c or 'wide' in c or 'haraimodoshi' in c]
for c in payout_cols:
    print(c)

print()
print("=== Sample data ===")
sample_query = """
SELECT 
    kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango,
    haraimodoshi_umaren_1a, haraimodoshi_umaren_1b,
    haraimodoshi_wide_1a, haraimodoshi_wide_1b,
    haraimodoshi_wide_2a, haraimodoshi_wide_2b,
    haraimodoshi_wide_3a, haraimodoshi_wide_3b
FROM jvd_hr
WHERE haraimodoshi_umaren_1b != '0' AND haraimodoshi_umaren_1b IS NOT NULL
LIMIT 5
"""
df = pd.read_sql(text(sample_query), e)
print(df)

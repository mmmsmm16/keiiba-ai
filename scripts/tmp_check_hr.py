from sqlalchemy import create_engine
import pandas as pd

e = create_engine("postgresql://postgres:postgres@host.docker.internal:5433/pckeiba")
df = pd.read_sql("""
SELECT 
    haraimodoshi_sanrenpuku_1a,
    haraimodoshi_sanrenpuku_1b,
    haraimodoshi_sanrenpuku_1c
FROM jvd_hr
WHERE kaisai_nen = '2025'
  AND haraimodoshi_sanrenpuku_1a IS NOT NULL
  AND haraimodoshi_sanrenpuku_1a != ''
LIMIT 5
""", e)
print(df)

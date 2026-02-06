"""Check for non-zero mining data"""
from sqlalchemy import create_engine, text
import pandas as pd

engine = create_engine('postgresql://postgres:postgres@db:5432/pckeiba')

# Check distribution of mining_kubun
print("=== Mining Kubun Distribution (2024) ===")
q1 = """
SELECT mining_kubun, COUNT(*) as cnt 
FROM jvd_se 
WHERE kaisai_nen = '2024' 
GROUP BY mining_kubun 
ORDER BY cnt DESC
"""
dist = pd.read_sql(text(q1), engine)
print(dist)

# Check for non-zero yoso_soha_time
print("\n=== Yoso Soha Time Sample (Non-zero) ===")
q2 = """
SELECT mining_kubun, yoso_soha_time, yoso_juni 
FROM jvd_se 
WHERE kaisai_nen = '2024' AND yoso_soha_time != '00000' 
LIMIT 10
"""
sample = pd.read_sql(text(q2), engine)
print(sample)

# Check how many rows have non-zero mining data
print("\n=== Data Availability ===")
q3 = """
SELECT COUNT(*) as total,
       SUM(CASE WHEN mining_kubun != '0' THEN 1 ELSE 0 END) as has_mining,
       SUM(CASE WHEN yoso_soha_time != '00000' THEN 1 ELSE 0 END) as has_yoso_time,
       SUM(CASE WHEN yoso_juni > 0 THEN 1 ELSE 0 END) as has_yoso_juni
FROM jvd_se 
WHERE kaisai_nen = '2024'
"""
avail = pd.read_sql(text(q3), engine)
print(avail)

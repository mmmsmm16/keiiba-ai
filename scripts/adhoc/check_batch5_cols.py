"""Check potential Batch 5 columns"""
import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine('postgresql://postgres:postgres@db:5432/pckeiba')

print("1. Mining Columns (jvd_se)")
query = "SELECT column_name FROM information_schema.columns WHERE table_name = 'jvd_se' AND (column_name LIKE '%mining%' OR column_name LIKE '%yoso%')"
cols = pd.read_sql(text(query), engine)
print(cols['column_name'].tolist())

print("\n2. Bloodline Columns (jvd_bt)")
query = "SELECT column_name FROM information_schema.columns WHERE table_name = 'jvd_bt'"
cols = pd.read_sql(text(query), engine)
print(cols['column_name'].tolist())

print("\n3. Sample Bloodline Descriptions")
query = "SELECT keito_id, keito_mei, keito_setsumei FROM jvd_bt LIMIT 5"
df = pd.read_sql(text(query), engine)
print(df)

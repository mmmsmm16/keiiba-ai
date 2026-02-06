"""Check jvd_hr column names"""
from sqlalchemy import create_engine, text
import pandas as pd

engine = create_engine('postgresql://postgres:postgres@db:5432/pckeiba')

# Check tansho columns
q1 = "SELECT column_name FROM information_schema.columns WHERE table_name = 'jvd_hr' AND column_name LIKE '%tansho%'"
cols = pd.read_sql(text(q1), engine)
print('Tansho columns:', cols['column_name'].tolist())

# Check haraimodoshi columns
q2 = "SELECT column_name FROM information_schema.columns WHERE table_name = 'jvd_hr' AND column_name LIKE 'haraimodoshi%' LIMIT 30"
cols2 = pd.read_sql(text(q2), engine)
print('\nAll haraimodoshi columns:')
for c in cols2['column_name'].tolist():
    print(f"  {c}")

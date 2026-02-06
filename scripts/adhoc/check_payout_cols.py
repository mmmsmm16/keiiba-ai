"""Check jvd_hr payout column structure"""
from sqlalchemy import create_engine, text

e = create_engine('postgresql://postgres:postgres@host.docker.internal:5433/pckeiba')
c = e.connect()

# Get columns
q = """
SELECT column_name 
FROM information_schema.columns 
WHERE table_name = 'jvd_hr' AND column_name LIKE 'haraimodoshi%' 
ORDER BY column_name
"""
r = c.execute(text(q))
cols = [row[0] for row in r]
print("Payout columns in jvd_hr:")
for col in cols:
    print(f"  {col}")

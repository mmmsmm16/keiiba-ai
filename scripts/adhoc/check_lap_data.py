import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine('postgresql://postgres:postgres@db:5432/pckeiba')

query = """
SELECT lap_time, zenhan_3f, kohan_3f, zenhan_4f, kohan_4f 
FROM jvd_ra 
WHERE zenhan_3f IS NOT NULL AND zenhan_3f != '000' 
LIMIT 10
"""

df = pd.read_sql(text(query), engine)
print("Sample Lap Data:")
print(df)
print("\nLap Time Sample (first row):", df['lap_time'].iloc[0] if not df.empty else "N/A")

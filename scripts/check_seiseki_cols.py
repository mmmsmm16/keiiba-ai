from sqlalchemy import create_engine
import pandas as pd
import os

db_url = "postgresql://postgres:postgres@localhost:5432/pckeiba"
engine = create_engine(db_url)

# Check columns of nvd_se (assuming it's a standard PC-KEIBA NAR table)
# We can try to query nvd_se or look at the schema
try:
    cols = pd.read_sql("SELECT * FROM nvd_se LIMIT 1", engine).columns.tolist()
    print("Columns in nvd_se:", cols)
except Exception as e:
    print(f"Error reading nvd_se: {e}")
    # Try alternate table name
    try:
        cols = pd.read_sql("SELECT * FROM nvd_seiseki LIMIT 1", engine).columns.tolist()
        print("Columns in nvd_seiseki:", cols)
    except Exception as e2:
        print(f"Error reading nvd_seiseki: {e2}")

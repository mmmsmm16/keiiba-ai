from sqlalchemy import create_engine
import pandas as pd
import os

user = os.environ.get('POSTGRES_USER', 'user')
password = os.environ.get('POSTGRES_PASSWORD', 'password')
host = os.environ.get('POSTGRES_HOST', 'db')
port = os.environ.get('POSTGRES_PORT', '5432')
dbname = os.environ.get('POSTGRES_DB', 'pckeiba')

connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
engine = create_engine(connection_str)

# Check columns of nvd_ra and nvd_se
tables = ['nvd_ra', 'nvd_se', 'nvd_race_shosai', 'nvd_seiseki']
for tbl in tables:
    try:
        df = pd.read_sql(f"SELECT * FROM {tbl} LIMIT 1", engine)
        print(f"Columns in {tbl}:", df.columns.tolist())
        # Print first row if exists
        if not df.empty:
            print(f"Sample data from {tbl}:")
            print(df.iloc[0])
    except Exception as e:
        print(f"Error reading {tbl}: {e}")

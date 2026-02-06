
import os
import pandas as pd
from sqlalchemy import create_engine

def main():
    user = os.environ.get('POSTGRES_USER', 'user')
    password = os.environ.get('POSTGRES_PASSWORD', 'password')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    
    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(connection_str)

    try:
        # Get unique course values and count
        query = "SELECT course, COUNT(*) as cnt FROM jvd_wc GROUP BY course ORDER BY cnt DESC LIMIT 20"
        df = pd.read_sql(query, engine)
        print("Unique courses in jvd_wc:")
        print(df)
        
        # Check first few rows for sample
        query_sample = "SELECT * FROM jvd_wc LIMIT 5"
        df_sample = pd.read_sql(query_sample, engine)
        print("\nSample rows:")
        print(df_sample[['course', 'babamawari', 'time_gokei_4f', 'lap_time_1f']])
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

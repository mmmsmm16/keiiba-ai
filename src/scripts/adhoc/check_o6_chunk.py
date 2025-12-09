
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
    
    query = "SELECT odds_sanrentan FROM jvd_o6 LIMIT 1"
    df = pd.read_sql(query, engine)
    val = df.iloc[0]['odds_sanrentan']
    
    print(f"Total: {len(str(val))}")
    # Print first 3 chunks of 17 bytes
    s = str(val)
    for i in range(3):
        chunk = s[i*17 : (i+1)*17]
        print(f"Chunk {i}: {chunk}")

if __name__ == "__main__":
    main()

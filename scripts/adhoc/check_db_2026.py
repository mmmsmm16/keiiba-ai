
import pandas as pd
from sqlalchemy import create_engine
import os

def check_db():
    user = os.environ.get('POSTGRES_USER', 'postgres')
    password = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    host = os.environ.get('POSTGRES_HOST', 'host.docker.internal')
    port = os.environ.get('POSTGRES_PORT', '5433')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    
    engine = create_engine(connection_str)
    
    # Check apd_sokuho_o1
    query = "SELECT kaisai_nen, COUNT(*) as cnt FROM apd_sokuho_o1 WHERE kaisai_nen >= '2025' GROUP BY kaisai_nen"
    df = pd.read_sql(query, engine)
    print("--- apd_sokuho_o1 count by year ---")
    print(df)
    
    # Check jvd_ra
    print("\n--- jvd_ra years count ---")
    query_ra = "SELECT kaisai_nen, COUNT(*) as cnt FROM jvd_ra WHERE kaisai_nen >= '2024' GROUP BY kaisai_nen"
    df_ra = pd.read_sql(query_ra, engine)
    print(df_ra)
    
    # Check if a specific 2026 race exists
    print("\n--- jvd_ra 2026 keibajo counts ---")
    query_v = "SELECT keibajo_code, COUNT(*) FROM jvd_ra WHERE kaisai_nen = '2026' GROUP BY keibajo_code"
    df_v = pd.read_sql(query_v, engine)
    print(df_v)

if __name__ == "__main__":
    check_db()

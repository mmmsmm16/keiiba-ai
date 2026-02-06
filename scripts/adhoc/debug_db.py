
import psycopg2
import pandas as pd

conn_str = "host='host.docker.internal' port=5433 dbname='pckeiba' user='postgres' password='postgres'"
try:
    conn = psycopg2.connect(conn_str)
    print("Connection successful!")
    q = "SELECT kaisai_nen, count(*) FROM jvd_hr GROUP BY kaisai_nen ORDER BY kaisai_nen DESC LIMIT 5"
    df = pd.read_sql(q, conn)
    print(df)
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")

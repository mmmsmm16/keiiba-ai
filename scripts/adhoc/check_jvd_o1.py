
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    print("Checking jvd_o1 table for 2026-01-18...")
    conn = psycopg2.connect(
        host='host.docker.internal',
        port=5433,
        dbname='pckeiba',
        user='postgres',
        password='postgres'
    )
    
    query = "SELECT * FROM jvd_o1 WHERE kaisai_nen = '2026' AND kaisai_tsukihi = '0118' LIMIT 5"
    df = pd.read_sql(query, conn)
    
    if df.empty:
        print("No records found in jvd_o1 for 2026-01-18.")
        
        # Check nearby dates
        print("Checking other dates in 2026...")
        query_all = "SELECT kaisai_nen, kaisai_tsukihi, count(*) FROM jvd_o1 WHERE kaisai_nen = '2026' GROUP BY kaisai_nen, kaisai_tsukihi ORDER BY kaisai_tsukihi LIMIT 10"
        df_all = pd.read_sql(query_all, conn)
        print(df_all)
    else:
        print(f"Found {len(df)} records.")
        print("Columns:", df.columns.tolist())
        print("Sample Data:")
        print(df[['kaisai_nen', 'kaisai_tsukihi', 'odds_tansho']].head())

    conn.close()

if __name__ == "__main__":
    main()

import pandas as pd
from sqlalchemy import create_engine
import os

def check():
    # Connect to Host DB (where user says data exists)
    url = 'postgresql://postgres:postgres@host.docker.internal:5433/pckeiba'
    engine = create_engine(url)
    
    # Check jvd_o1 (Odds) for recent/future dates
    query = "SELECT kaisai_nen, kaisai_tsukihi, count(*) FROM jvd_o1 WHERE kaisai_nen='2025' AND kaisai_tsukihi >= '1214' GROUP BY kaisai_nen, kaisai_tsukihi ORDER BY kaisai_tsukihi DESC"
    df = pd.read_sql(query, engine)
    print("Latest Odds Dates in DB (jvd_o1):")
    print(df)

if __name__ == "__main__":
    check()

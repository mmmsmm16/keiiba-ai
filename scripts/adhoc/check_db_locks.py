
import os
import pandas as pd
from sqlalchemy import create_engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    user = os.environ.get('POSTGRES_USER', 'user')
    password = os.environ.get('POSTGRES_PASSWORD', 'password')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    connection_str = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    
    engine = create_engine(connection_str)
    
    query = """
    SELECT pid, usename, state, query_start, wait_event_type, wait_event, query
    FROM pg_stat_activity
    WHERE state != 'idle'
    ORDER BY query_start
    """
    
    try:
        df = pd.read_sql(query, engine)
        print(df)
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()

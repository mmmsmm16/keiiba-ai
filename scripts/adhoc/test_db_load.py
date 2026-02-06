
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
    
    logger.info("Connecting to DB...")
    engine = create_engine(connection_str)
    
    query = "SELECT * FROM jvd_ra LIMIT 100"
    logger.info(f"Executing query: {query}")
    
    try:
        df = pd.read_sql(query, engine)
        logger.info(f"Loaded {len(df)} rows.")
        print(df.head())
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()

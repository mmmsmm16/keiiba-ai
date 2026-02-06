import os
import pandas as pd
from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    user = os.environ.get('POSTGRES_USER', 'user')
    password = os.environ.get('POSTGRES_PASSWORD', 'password')
    host = os.environ.get('POSTGRES_HOST', 'db')
    port = os.environ.get('POSTGRES_PORT', '5432')
    dbname = os.environ.get('POSTGRES_DB', 'pckeiba')
    db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(db_url)
    
    logger.info("Checking apd_sokuho_o1...")
    
    try:
        # Check count for 2024
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM apd_sokuho_o1 WHERE kaisai_nen = '2024'")).scalar()
            logger.info(f"Records in 2024: {count}")
            
            if count > 0:
                # Sample one race
                sample = pd.read_sql(text("SELECT * FROM apd_sokuho_o1 WHERE kaisai_nen = '2024' LIMIT 1"), conn)
                logger.info(f"Sample Columns: {sample.columns.tolist()}")
                logger.info(f"Sample Data:\n{sample.iloc[0]}")
                
            # Test T-10 Fetch Logic efficiency
            # Strategy: Join with jvd_ra to get start time, then find latest odds <= start - 10min
            # Using LATERAL JOIN or DISTINCT ON
            logger.info("Testing T-10 Query efficiency (2024, Limit 10 races)...")
            
            # Note: Timestamp construction in SQL might be tricky with strings
            # jvd_ra.hasso_jikoku is '1540' (HHMM). date is 'MMDD'.
            # apd_sokuho_o1.happyo_tsukihi_jifun is 'MMDDHHMM'.
            
            query = """
            WITH target_races AS (
                SELECT 
                    kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango,
                    hasso_jikoku
                FROM jvd_ra
                WHERE kaisai_nen = '2024' AND data_kubun = '7'
                LIMIT 10
            )
            SELECT
                r.race_bango,
                r.hasso_jikoku,
                o1.happyo_tsukihi_jifun,
                o1.odds_tansho
            FROM target_races r
            LEFT JOIN LATERAL (
                SELECT * FROM apd_sokuho_o1 o
                WHERE o.kaisai_nen = r.kaisai_nen
                  AND o.keibajo_code = r.keibajo_code
                  AND o.kaisai_kai = r.kaisai_kai
                  AND o.kaisai_nichime = r.kaisai_nichime
                  AND o.race_bango = r.race_bango
                  -- Time Comparison Logic (Simplified for check)
                  -- AND o.happyo_tsukihi_jifun <= (TargetTime)
                ORDER BY o.happyo_tsukihi_jifun DESC
                LIMIT 1
            ) o1 ON TRUE
            """
            
            df = pd.read_sql(text(query), conn)
            logger.info(f"Fetched {len(df)} rows.")
            print(df)

    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()

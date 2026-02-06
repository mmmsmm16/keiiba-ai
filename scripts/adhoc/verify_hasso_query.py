
import os
import sys
import pandas as pd
import logging
from dotenv import load_dotenv

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from src.preprocessing.loader import JraVanDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    load_dotenv()
    target_date = "20251221"
    logger.info(f"Testing SQL query for {target_date}...")
    
    loader = JraVanDataLoader()
    
    # The updated query
    query_se = f"""
        SELECT se.*, bamei, kishumei_ryakusho as kishu_mei, tansho_odds as odds_tansho, ra.hasso_jikoku
        FROM jvd_se se
        LEFT JOIN jvd_ra ra ON 
            se.kaisai_nen = ra.kaisai_nen AND
            se.keibajo_code = ra.keibajo_code AND
            se.kaisai_kai = ra.kaisai_kai AND
            se.kaisai_nichime = ra.kaisai_nichime AND
            se.race_bango = ra.race_bango
        WHERE se.kaisai_nen = '{target_date[:4]}' 
        AND se.kaisai_tsukihi = '{target_date[4:]}'
        AND se.keibajo_code BETWEEN '01' AND '10'
    """
    
    try:
        df = pd.read_sql(query_se, loader.engine)
        if df.empty:
            logger.warning("No data found!")
        else:
            logger.info(f"Found {len(df)} rows.")
            logger.info("Sample columns: " + str(df.columns.tolist()))
            
            # Check hasso_jikoku
            if 'hasso_jikoku' in df.columns:
                logger.info("hasso_jikoku validation:")
                print(df[['kaisai_nen', 'keibajo_code', 'race_bango', 'hasso_jikoku']].head(10))
                
                # Check sortability
                df['hasso_jikoku'] = df['hasso_jikoku'].fillna('9999').astype(str)
                df_sorted = df.sort_values('hasso_jikoku')
                print("Sorted sample:")
                print(df_sorted[['race_bango', 'hasso_jikoku']].head(5))
            else:
                logger.error("hasso_jikoku column missing!")
                
    except Exception as e:
        logger.error(f"Query failed: {e}")

if __name__ == "__main__":
    main()

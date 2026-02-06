
import os
import sys
import pandas as pd
import logging
from pprint import pprint
from dotenv import load_dotenv

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from src.preprocessing.loader import JraVanDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    load_dotenv()
    
    # 2024 Arima Kinen (Dec 22, 2024, Nakayama 11R)
    # race_id might be 202406050811 (Example ID formation, need to be careful)
    # Let's search by date directly.
    target_date = "1222" # Dec 22
    target_year = "2024"
    
    logger.info(f"Checking Time-Series Odds for {target_year}-{target_date}...")
    
    loader = JraVanDataLoader()
    
    # Select specific columns to avoid data type issues
    query_ts = """
        SELECT happyo_tsukihi_jifun, odds_tansho
        FROM apd_sokuho_o1
        WHERE kaisai_nen = :year
        AND happyo_tsukihi_jifun LIKE :date_pattern
        AND keibajo_code = '06'
        AND race_bango = '11'
        ORDER BY happyo_tsukihi_jifun ASC
        LIMIT 5
    """
    
    from sqlalchemy import text
    try:
        logger.info("Executing Time-Series Query (Direct SQLAlchemy)...")
        with loader.engine.connect() as conn:
            result = conn.execute(
                text(query_ts), 
                {"year": target_year, "date_pattern": f"%{target_date}%"}
            )
            rows = result.fetchall()
            
            if not rows:
                logger.warning("No time-series odds data found for the target race.")
            else:
                logger.info(f"Found {len(rows)} records.")
                print("\n=== Sample Time-Series Data (apd_sokuho_o1) ===")
                for row in rows:
                    print(row)
                    
                # Inspect the first row data types
                first_row = rows[0]
                print(f"Data Types: {[type(c) for c in first_row]}")

    except Exception as e:
        logger.error(f"Error querying apd_sokuho_o1: {e}")
        
    # Check Final Odds (jvd_se) for comparison
    query_final = f"""
        SELECT *
        FROM jvd_se
        WHERE kaisai_nen = '{target_year}'
        AND kaisai_tsukihi = '{target_date}'
        AND keibajo_code = '06'
        AND race_bango = '11'
    """
    try:
        logger.info("Executing Final Odds Query...")
        df_final = pd.read_sql(query_final, loader.engine)
        if not df_final.empty:
             print("\n=== Final Odds Data (jvd_se) ===")
             print(df_final[['umaban', 'tansho_odds']].sort_values('umaban').head(5))
    except Exception as e:
        logger.error(f"Error querying jvd_se: {e}")

if __name__ == "__main__":
    main()

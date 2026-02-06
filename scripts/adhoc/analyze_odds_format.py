
import os
import sys
import pandas as pd
import logging
from sqlalchemy import text
from dotenv import load_dotenv

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from src.preprocessing.loader import JraVanDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    load_dotenv()
    target_date = "1222" # Dec 22
    target_year = "2024"
    
    loader = JraVanDataLoader()
    
    # 1. Inspect happyo_tsukihi_jifun distribution
    # Just list all timestamps for Nakayama 11R
    query_time = """
        SELECT happyo_tsukihi_jifun
        FROM apd_sokuho_o1
        WHERE kaisai_nen = :year
        AND keibajo_code = '06'
        AND race_bango = '11'
        ORDER BY happyo_tsukihi_jifun ASC
    """
    
    logger.info("Analyzing Timestamps...")
    try:
        with loader.engine.connect() as conn:
            result = conn.execute(text(query_time), {"year": target_year})
            rows = result.fetchall()
            
            logger.info(f"Found {len(rows)} timestamps.")
            for i, r in enumerate(rows):
                if i < 5 or i > len(rows) - 5:
                    print(f"  {r[0]}")
    except Exception as e:
        logger.error(f"Timestamp query failed: {e}")

    # 2. Analyze Odds String Logic
    # 06 Nakayama 11R Arima Kinen, Horse 1 should be '01', Horse 10 should be '10'.
    # Win Odds for Horse 1 was 4.0? Let's check final again.
    # Previous output of final odds: 01->4.0(0040), 02->?(0000?), 03->2.8(0028)...
    # Wait, 0040 is 4.0. 0028 is 2.8.
    
    # Let's get ONE full odds string and try to parse it
    query_str = """
        SELECT happyo_tsukihi_jifun, odds_tansho
        FROM apd_sokuho_o1
        WHERE kaisai_nen = :year
        AND keibajo_code = '06'
        AND race_bango = '11'
        ORDER BY happyo_tsukihi_jifun DESC
        LIMIT 1
    """
    
    logger.info("\nAnalyzing Odds String (Latest)...")
    try:
        with loader.engine.connect() as conn:
            result = conn.execute(text(query_str), {"year": target_year})
            row = result.fetchone()
            
            if row:
                ts, odds_str = row
                print(f"Timestamp: {ts}")
                print(f"String Len: {len(odds_str)}")
                print(f"String: {odds_str}")
                
                # Parsing Trial
                # Assuming format: HorseNum(2) + Odds(4) + Popularity(2) = 8 chars per horse?
                # JRA-VAN Spec for Tansho (O1 record):
                # Data repeats for 18 horses (max).
                # Each horse:
                #   Umaban (2)
                #   Tansho Odds (4) - 9999 means cancel? or missing?
                #   Ninki (2)
                # Let's try to split by 8 chars
                
                print("\nParsing Trial (Assuming 8 chars chunk: Umaban(2)+Odds(4)+Ninki(2)):")
                chunk_size = 8
                for i in range(0, len(odds_str), chunk_size):
                    chunk = odds_str[i:i+chunk_size]
                    if len(chunk) < chunk_size: break
                    if chunk.strip() == "": break
                    
                    umaban = chunk[0:2]
                    odds_raw = chunk[2:6]
                    ninki = chunk[6:8]
                    
                    try:
                        odds_val = float(odds_raw) / 10.0
                        print(f"  Horse {umaban}: Odds {odds_val:>5.1f} (Raw: {odds_raw}), Rank {ninki}")
                    except:
                        print(f"  Horse {umaban}: Parse Error ({chunk})")
                        
    except Exception as e:
        logger.error(f"Odds string query failed: {e}")

if __name__ == "__main__":
    main()

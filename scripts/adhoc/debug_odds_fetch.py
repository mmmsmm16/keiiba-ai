
import sys
import os
import logging
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from preprocessing.loader import JraVanDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    loader = JraVanDataLoader()
    
    # Query a small sample of recent races
    # Query JRA races (01-10) to check format and odds
    query = """
    SELECT 
        kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
        kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango,
        umaban as horse_number,
        tansho_odds 
    FROM jvd_se 
    WHERE kaisai_nen = '2023' AND keibajo_code IN ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10')
    LIMIT 20
    """
    try:
        df = pd.read_sql(query, loader.engine)
        print("Raw JRA Data Sample:")
        print(df.head())
        
        # Check race_id format
        if not df.empty:
            sample_race_id = df.iloc[0]['race_id']
            print(f"Sample Race ID: '{sample_race_id}' (Len: {len(sample_race_id)})")
            
            # Check components length
            print(f"Components: {df.iloc[0][['kaisai_nen', 'keibajo_code', 'kaisai_kai', 'kaisai_nichime', 'race_bango']].tolist()}")
        
        # Check odds parsing
        if not df.empty:
            df['odds_numeric'] = pd.to_numeric(df['tansho_odds'], errors='coerce')
            print("Numeric Odds Sample:")
            print(df[['tansho_odds', 'odds_numeric']].head())
            print(f"Mean Odds: {df['odds_numeric'].mean()}")
        
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()

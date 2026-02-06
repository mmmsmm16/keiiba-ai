
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"

def inspect():
    logger.info(f"Loading {DATA_PATH}...")
    try:
        df = pd.read_parquet(DATA_PATH)
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns (First 20): {df.columns.tolist()[:20]}")
        
        target_cols = ['rank', 'is_win', 'odds', 'date', 'race_id']
        found = [c for c in target_cols if c in df.columns]
        missing = [c for c in target_cols if c not in df.columns]
        
        logger.info(f"Found targets: {found}")
        logger.info(f"Missing targets: {missing}")
        
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    inspect()

import pandas as pd
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
import logging
import warnings

# Add workspace to path
sys.path.append('/workspace')
from src.preprocessing.loader import JraVanDataLoader
# from src.preprocessing.pipeline import FeaturePipeline # This was wrong class
from src.preprocessing.feature_pipeline import FeaturePipeline
from src.preprocessing.features.odds_fluctuation import compute_odds_fluctuation

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Configuration
CACHE_DIR = "data/features_v13/rebuild_active"
OUTPUT_FILE = "data/processed/preprocessed_data_v13_active.parquet"
HISTORY_START = '2019-01-01'
N_WORKERS = 4  # Adjust based on CPU

def process_year(year):
    logger.info(f"Processing Year: {year}")
    loader = JraVanDataLoader()
    
    # Load Raw Data
    try:
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        logger.info(f"Loading raw data for {year}...")
        df_raw = loader.load(history_start_date=start_date, end_date=end_date, skip_odds=False)
        
        if df_raw.empty:
            logger.warning(f"No data for {year}")
            return None
            
        logger.info(f"Loaded {len(df_raw)} rows for {year}")
        
        # --- MANUAL ODDS FLUCTUATION INJECTION ---
        logger.info("Manually computing odds fluctuation...")
        df_odds = compute_odds_fluctuation(df_raw)
        if not df_odds.empty:
            logger.info(f"Odds Fluctuation computed: {len(df_odds)} rows. Merging...")
            # drop duplicates if any
            df_odds = df_odds.drop_duplicates(subset=['race_id', 'horse_number'])
            
            # Ensure keys match (str vs int)
            df_raw['race_id'] = df_raw['race_id'].astype(str)
            df_raw['horse_number'] = df_raw['horse_number'].astype(int)
            
            # Merge
            df_raw = pd.merge(df_raw, df_odds.drop(columns=['horse_id'], errors='ignore'), 
                              on=['race_id', 'horse_number'], how='left')
            logger.info("Odds merged.")
        else:
            logger.warning("compute_odds_fluctuation returned empty!")
        # ----------------------------------------
        
        # Initialize Pipeline
        pipeline = FeaturePipeline(cache_dir=CACHE_DIR)
        
        # We need ALL blocks
        blocks = list(pipeline.registry.keys())
        
        # If we manually computed odds, we might want to skip 'odds_fluctuation' block in pipeline to avoid overwrite?
        # But 'odds_fluctuation' block logic is same. If it fails there, it might overwrite with NaNs?
        # It depends on how pipeline merges.
        # FeaturePipeline usually merges output to df.
        # If output is empty DF, it does nothing?
        # Let's remove 'odds_fluctuation' from blocks just in case.
        if 'odds_fluctuation' in blocks:
            blocks.remove('odds_fluctuation')
            
        logger.info(f"Generating features for {year} with {len(blocks)} blocks...")
        df_features = pipeline.load_features(df_raw, blocks)
        
        # Attach Target Columns (Rank, Final Odds, etc for Training Target)
        # Note: 'odds' column in df_raw is Final Odds. We keep it for Target label, but will NOT use it for Feature.
        target_cols = ['rank', 'odds', 'popularity', 'time', 'date', 'race_id']
        # Merge target cols if not present
        existing_cols = df_features.columns
        for col in target_cols:
            if col not in existing_cols and col in df_raw.columns:
                df_features[col] = df_raw[col]
             
        # Ensure manually merged odds are in final output!
        # Pipeline usually returns NEW DF with only features?
        # FeaturePipeline.load_features usually accumulates features.
        # Does it preserve input columns?
        # Check src/preprocessing/feature_pipeline.py or assume we need to re-attach.
        # IF load_features return only computed features, we lose our manual odds!
        # SO we must attach manual odds columns to df_features if missing.
        manual_odds_cols = ['odds_10min', 'odds_ratio_10min', 'odds_rank_10min']
        for c in df_raw.columns:
            if 'odds' in c and c not in df_features.columns:
                df_features[c] = df_raw[c]
                
        return df_features
        
    except Exception as e:
        logger.error(f"Error processing {year}: {e}")
        return None

def main():
    years = [2024, 2025] # Recent 2 years for V14 MVP (Rapid Rebuild)
    
    dfs = []
    
    print("Starting Dataset Rebuild V13 (Active / No Leak) - Sequential...")
    
    for year in years:
        try:
            df = process_year(year)
            if df is not None:
                dfs.append(df)
        except Exception as e:
            logger.error(f"Failed year {year}: {e}")
            
    if not dfs:
        logger.error("No data generated.")
        return
        
    logger.info("Concatenating years...")
    df_all = pd.concat(dfs, ignore_index=True)
    
    # Sort
    df_all = df_all.sort_values('date').reset_index(drop=True)
    
    # Save
    logger.info(f"Saving to {OUTPUT_FILE}...")
    df_all.to_parquet(OUTPUT_FILE)
    logger.info("Done.")

if __name__ == "__main__":
    main()

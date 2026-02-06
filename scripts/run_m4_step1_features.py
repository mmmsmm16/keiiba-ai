
import sys
import os
import logging
import pandas as pd
import numpy as np
import traceback
import gc

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from preprocessing.loader import JraVanDataLoader
from preprocessing.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Starting M4 Step 1: Feature Generation")
    
    # Config
    START_DATE = "2023-01-01"
    END_DATE = "2024-12-31"
    
    FEATURE_BLOCKS = [
        'base_attributes', 'history_stats', 'jockey_stats', 
        'temporal_jockey_stats', 'temporal_trainer_stats',
        'burden_stats', 'changes_stats', 'aptitude_stats', 
        # 'speed_index_stats', # Crash
        'pace_pressure_stats',
        'relative_stats' # Re-enable relative stats if possible, or keep pruned if still crash
    ]
    # Pruned version as per latest ad-hoc state:
    FEATURE_BLOCKS = [
        'base_attributes', 'history_stats', 'jockey_stats', 
        'temporal_jockey_stats', 'temporal_trainer_stats',
        'burden_stats', 'changes_stats', 'aptitude_stats', 
        # 'speed_index_stats', 
        # 'pace_pressure_stats', 
        # 'relative_stats'
    ]
    # Restore M4-A Baseline (Core) config
    FEATURE_BLOCKS = [
        'base_attributes', 'history_stats', 'jockey_stats', 
        'temporal_jockey_stats', 'temporal_trainer_stats',
        'burden_stats', 'changes_stats', 'aptitude_stats', 
    ]
    
    os.makedirs("data/temp_m4", exist_ok=True)
    
    # 1. Data Loading (Resume Logic)
    logger.info(f"Loading Data ({START_DATE} ~ {END_DATE})...")
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    temp_files = []
    
    current_year = start_dt.year
    final_year = end_dt.year
    
    while current_year <= final_year:
        gc.collect()
        y_start = f"{current_year}-01-01"
        y_end = f"{current_year}-12-31"
        if current_year == start_dt.year: y_start = START_DATE
        if current_year == final_year: y_end = END_DATE
        
        temp_file = f"data/temp_m4/year_{current_year}.parquet"
        if os.path.exists(temp_file):
            logger.info(f"  Found cached data for {current_year}: {temp_file}")
            temp_files.append(temp_file)
            current_year += 1
            continue
            
        logger.info(f"  Loading Year {current_year} ({y_start} ~ {y_end})...")
        try:
            loader = JraVanDataLoader() 
            df_year = loader.load(limit=None, history_start_date=y_start, end_date=y_end, skip_odds=True, skip_training=True)
            if df_year is not None and len(df_year) > 0:
                df_year.to_parquet(temp_file)
                temp_files.append(temp_file)
                logger.info(f"    -> Saved {len(df_year)} rows to {temp_file}")
                del df_year
                del loader
                gc.collect()
            else:
                logger.warning(f"    -> No data for {current_year}")
        except Exception as e:
            logger.error(f"    -> Failed to load {current_year}: {e}")
            traceback.print_exc()
        current_year += 1
        
    if not temp_files:
        logger.error("No data loaded!")
        return

    logger.info("Concatenating yearly data from disk...")
    dfs = [pd.read_parquet(f) for f in temp_files]
    raw_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total Loaded Raw Rows: {len(raw_df)}")
    del dfs
    gc.collect()
    
    # 2. Cleansing
    logger.info("Cleansing Data (INLINE MINIMAL)...")
    clean_df = raw_df.copy()
    if 'rank' in clean_df.columns:
        clean_df['rank'] = pd.to_numeric(clean_df['rank'], errors='coerce').fillna(0).astype(int)
        clean_df = clean_df[clean_df['rank'] != 0].copy()
    clean_df['time_diff'] = np.nan
    logger.info(f"Cleansing Complete. Shape: {clean_df.shape}")
    del raw_df
    gc.collect()
    
    # 3. Save Targets
    logger.info("Saving Targets for Step 2...")
    target_cols = ['race_id', 'horse_number', 'rank', 'date']
    if 'date' not in clean_df.columns and 'race_id' in clean_df.columns:
        # Check if date is missing (it shouldn't be in raw_df)
        pass
    
    clean_df[target_cols].to_parquet("data/temp_m4/M4_targets.parquet")
    logger.info("Targets saved to data/temp_m4/M4_targets.parquet")
    
    # 4. Feature Pipeline
    logger.info("Generating Features...")
    pipeline = FeaturePipeline(cache_dir="data/features")
    df_features = pipeline.load_features(clean_df, FEATURE_BLOCKS)
    logger.info(f"Features Ready: {df_features.shape}")
    
    # Save Features
    logger.info("Saving Features for Step 2...")
    df_features.to_parquet("data/temp_m4/M4_features.parquet")
    logger.info("Features saved to data/temp_m4/M4_features.parquet")
    
    logger.info("Step 1 Complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.error("🔥 Fatal Error in Step 1:")
        traceback.print_exc()
        sys.exit(1)

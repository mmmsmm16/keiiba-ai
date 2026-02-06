
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
    logger.info("🚀 Starting Phase Q Step 1: Feature Generation (Class Stats Enabled)")
    
    # Config
    START_DATE = "2015-01-01"
    END_DATE = "2024-12-31"
    
    # Feature Set including Class Stats
    FEATURE_BLOCKS = [
        'base_attributes', 'history_stats', 'jockey_stats', 
        'pace_stats', 'bloodline_stats', 'training_stats',
        'burden_stats', 'changes_stats', 'aptitude_stats', 
        'speed_index_stats', 'pace_pressure_stats',
        'relative_stats', 'jockey_trainer_stats',
        'class_stats' # New Feature
    ]
    
    TEMP_DIR = "data/temp_q1"
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 1. Data Loading (Reuse M5 temp if available, otherwise M5 logic)
    # Actually, we can just look for data/temp_m5 data to save time?
    # No, let's just implement the standard loader logic to be safe and independent.
    
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
        
        # Reuse M5 temp for speed if possible
        m5_temp = f"data/temp_m5/year_{current_year}.parquet"
        q1_temp = f"{TEMP_DIR}/year_{current_year}.parquet"
        
        m5_temp = f"data/temp_m5/year_{current_year}.parquet"
        # if os.path.exists(m5_temp):
        #      logger.info(f"  Reuse M5 cached data for {current_year}: {m5_temp}")
        #      temp_files.append(m5_temp)
        #      current_year += 1
        #      continue
        
        if os.path.exists(q1_temp):
            logger.info(f"  Found cached data for {current_year}: {q1_temp}")
            temp_files.append(q1_temp)
            current_year += 1
            continue
            
        logger.info(f"  Loading Year {current_year} ({y_start} ~ {y_end})...")
        try:
            loader = JraVanDataLoader() 
            df_year = loader.load(limit=None, history_start_date=y_start, end_date=y_end, skip_odds=True, skip_training=False)
            if df_year is not None and len(df_year) > 0:
                df_year.to_parquet(q1_temp)
                temp_files.append(q1_temp)
                logger.info(f"    -> Saved {len(df_year)} rows to {q1_temp}")
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

    logger.info("Concatenating yearly data...")
    dfs = [pd.read_parquet(f) for f in temp_files]
    raw_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total Loaded Raw Rows: {len(raw_df)}")
    del dfs
    gc.collect()
    
    # 2. Cleansing
    logger.info("Cleansing Data...")
    clean_df = raw_df.copy()
    if 'rank' in clean_df.columns:
        clean_df['rank'] = pd.to_numeric(clean_df['rank'], errors='coerce').fillna(0).astype(int)
        clean_df = clean_df[clean_df['rank'] != 0].copy()
    
    # [Fix] Ensure last_3f is numeric (It was causing string comparison errors in pace_stats)
    if 'last_3f' in clean_df.columns:
        clean_df['last_3f'] = pd.to_numeric(clean_df['last_3f'], errors='coerce') / 10.0

    clean_df['time_diff'] = np.nan
    del raw_df
    gc.collect()
    
    # 3. Save Targets
    target_cols = ['race_id', 'horse_number', 'rank', 'date']
    clean_df[target_cols].to_parquet(f"{TEMP_DIR}/Q1_targets.parquet")
    logger.info(f"Targets saved to {TEMP_DIR}/Q1_targets.parquet")
    
    # 4. Feature Pipeline
    logger.info(f"Generating Features with blocks: {FEATURE_BLOCKS}")
    pipeline = FeaturePipeline(cache_dir="data/features_q1")
    df_features = pipeline.load_features(clean_df, FEATURE_BLOCKS)
    logger.info(f"Features Ready: {df_features.shape}")
    
    # Verify Class Stats Presence
    if 'hc_n_races_365d' not in df_features.columns:
         logger.error("❌ CRITICAL: class_stats features missing!")
         raise ValueError("Feature generation failed to include class_stats")
    
    # Save Features
    df_features.to_parquet(f"{TEMP_DIR}/Q1_features.parquet")
    logger.info(f"Features saved to {TEMP_DIR}/Q1_features.parquet")
    
    logger.info("Phase Q Step 1 Complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.error("🔥 Fatal Error in Step 1:")
        traceback.print_exc()
        sys.exit(1)

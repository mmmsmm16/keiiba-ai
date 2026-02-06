
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
    
    TEMP_DIR = "data/temp_q2"
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
    years = range(current_year, final_year + 1)
    
    for current_year in years:
        gc.collect()
        y_start = f"{current_year}-01-01"
        y_end = f"{current_year}-12-31"
        if current_year == start_dt.year: y_start = START_DATE
        if current_year == end_dt.year: y_end = END_DATE
        
        logger.info(f"Processing Year {current_year}...")
        
        # Check Cache
        q2_temp = f"{TEMP_DIR}/year_{current_year}.parquet"
        
        # Reuse Q1 data if available? 
        # Actually Q1 is already cleansed. But we need features, not raw data here.
        # This script merges features.
        # Wait, this script LOADS raw data then PIPELINE features.
        # We can reuse Q1 temp raw data if we want, but it's cleaner to use Q1 features cache we copied.
        # But here we need RAW data to pass to pipeline.
        
        # Actually, Q1 temp raw files are in data/temp_q1. We can reuse them to skip DB load.
        # Let's check q1 temp first.
        q1_temp = f"data/temp_q1/year_{current_year}.parquet"
        
        if os.path.exists(q1_temp):
             logger.info(f"  Reuse Q1 cached raw data for {current_year}: {q1_temp}")
             temp_files.append(q1_temp)
             continue
        
        if os.path.exists(q2_temp):
            logger.info(f"  Found cached data for {current_year}: {q2_temp}")
            temp_files.append(q2_temp)
            continue
            
        logger.info(f"  Loading Year {current_year} ({y_start} ~ {y_end})...")
        try:
            loader = JraVanDataLoader() 
            df_year = loader.load(limit=None, history_start_date=y_start, end_date=y_end, skip_odds=True, skip_training=False)
            if df_year is not None and len(df_year) > 0:
                # 2. Cleansing (moved inside loop for yearly processing)
                logger.info(f"  Cleansing Data for Year {current_year}...")
                if 'rank' in df_year.columns:
                    df_year['rank'] = pd.to_numeric(df_year['rank'], errors='coerce').fillna(0).astype(int)
                    df_year = df_year[df_year['rank'] != 0].copy()
                
                # [Fix] Ensure last_3f is numeric (It was causing string comparison errors in pace_stats)
                if 'last_3f' in df_year.columns:
                    df_year['last_3f'] = pd.to_numeric(df_year['last_3f'], errors='coerce') / 10.0

                df_year['time_diff'] = np.nan

                df_year.to_parquet(q2_temp)
                temp_files.append(q2_temp)
                logger.info(f"    -> Saved {len(df_year)} rows to {q2_temp}")
                del df_year
                del loader
                gc.collect()
            else:
                logger.warning(f"    -> No data for {current_year}")
        except Exception as e:
            logger.error(f"    -> Failed to load {current_year}: {e}")
            traceback.print_exc()
        
    if not temp_files:
        logger.error("No data loaded!")
        return

    # 3. Concatenate
    logger.info("Concatenating yearly data...")
    # Cleanse again? No, loaded data is already cleansed?
    # q1/temp files are saved AFTER processing (lines 98-99 in original).
    # So we just concat.
    
    clean_df = pd.concat([pd.read_parquet(f) for f in temp_files], ignore_index=True)
    
    logger.info(f"Total Loaded Raw Rows: {len(clean_df)}")
    
    # Save targets (Q2)
    target_cols = ['race_id', 'horse_number', 'rank', 'date']
    clean_df[target_cols].to_parquet("data/temp_q2/Q2_targets.parquet")
    logger.info("Targets saved to data/temp_q2/Q2_targets.parquet")
    
    # 4. Feature Pipeline
    logger.info(f"Generating Features with blocks: {FEATURE_BLOCKS}")
    pipeline = FeaturePipeline(cache_dir="data/features_q2")
    df_features = pipeline.load_features(clean_df, FEATURE_BLOCKS)
    logger.info(f"Features Ready: {df_features.shape}")
    
    # Verify Class Stats Presence
    if 'hc_n_races_365d' not in df_features.columns:
         logger.error("❌ CRITICAL: class_stats features missing!")
         raise ValueError("Feature generation failed to include class_stats")
    
    # Save Features
    out_path = f"{TEMP_DIR}/Q2_features.parquet"
    df_features.to_parquet(out_path)
    logger.info(f"Features saved to {out_path}")
    
    logger.info("Phase Q Step 1 Complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.error("🔥 Fatal Error in Step 1:")
        traceback.print_exc()
        sys.exit(1)

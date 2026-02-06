
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
    logger.info("🚀 Starting Phase Q Step 1 (Iteration 8): Interval Aptitude Feature Generation")
    
    # Feature Set (Base Q7 + Interval Aptitude)
    FEATURE_BLOCKS = [
        'base_attributes', 'history_stats', 'jockey_stats', 
        'pace_stats', 'bloodline_stats', 'training_stats',
        'burden_stats', 'changes_stats', 'aptitude_stats', 
        'speed_index_stats', 'pace_pressure_stats',
        'relative_stats', 'jockey_trainer_stats',
        'class_stats',
        'risk_stats',
        'course_aptitude',
        'extended_aptitude',
        'runstyle_fit',
        'jockey_trainer_compatibility',
        'interval_aptitude' # New
    ]
    
    # We use data/features_q8 as cache dir
    CACHE_DIR = "data/features_q8"
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # 1. Copy Cache from Q7 to Q8
    logger.info("Skipping Q7 cache copy to force regeneration for 2025...")
    
    # q7_cache = "data/features_q7"
    # if os.path.exists(q7_cache):
    #     import shutil
    #     import glob
    #     files = glob.glob(f"{q7_cache}/*.parquet")
    #     for f in files:
    #         bn = os.path.basename(f)
    #         dst = os.path.join(CACHE_DIR, bn)
    #         if not os.path.exists(dst):
    #              shutil.copy2(f, dst)
    #     logger.info("  -> Copied Q7 cache.")
    # else:
    #     logger.warning("  -> Q7 cache not found! Starting fresh.")
        
    # 2. Data Loading (Reuse Q7 Temp) - Actually same temp_q1 base.
    # But we define TEMP_DIR for Q8 outputs
    TEMP_DIR = "data/temp_q8"
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    q1_target_path = "data/temp_q1/Q1_targets.parquet" # Base targets
    q1_temp_raw_base = "data/temp_q1/year_" # year_{year}.parquet
    
    if not os.path.exists(q1_target_path):
        logger.error("Q1 Targets not found.")
        return

    # Load All Years
    logger.info("Loading Raw Data...")
    
    years = range(2015, 2026)
    raw_dfs = []
    for y in years:
        fpath = f"{q1_temp_raw_base}{y}.parquet"
        if os.path.exists(fpath):
            raw_dfs.append(pd.read_parquet(fpath))
            
    if not raw_dfs:
        logger.error("No raw data found!")
        return
        
    clean_df = pd.concat(raw_dfs, ignore_index=True)
    logger.info(f"Loaded {len(clean_df)} rows.")
    
    # Save Q8 Targets
    target_cols = ['race_id', 'horse_number', 'rank', 'date']
    clean_df[target_cols].to_parquet(f"{TEMP_DIR}/Q8_targets.parquet")
    
    # Pre-calculate time_diff if missing (Needed for history_stats)
    if 'time_diff' not in clean_df.columns:
        if 'time' in clean_df.columns:
            logger.info("Computing time_diff and correcting types...")
            clean_df['time'] = pd.to_numeric(clean_df['time'], errors='coerce')
            clean_df['last_3f'] = pd.to_numeric(clean_df.get('last_3f'), errors='coerce') # Ensure last_3f is numeric
            
            min_time = clean_df.groupby('race_id')['time'].transform('min')
            clean_df['time_diff'] = clean_df['time'] - min_time
            clean_df['time_diff'] = clean_df['time_diff'].fillna(99.9) 
        else:
             logger.warning("time column missing, cannot compute time_diff.")
             clean_df['time_diff'] = 0.0

    # 3. Feature Pipeline
    logger.info(f"Generating Features...")
    pipeline = FeaturePipeline(cache_dir=CACHE_DIR)
    
    # Force re-compute of new block
    new_block = "interval_aptitude"
    new_cache = os.path.join(CACHE_DIR, f"{new_block}.parquet")
    if os.path.exists(new_cache):
        os.remove(new_cache)
        
    df_features = pipeline.load_features(clean_df, FEATURE_BLOCKS)
    
    # Verify
    if 'tataki_count' not in df_features.columns:
         logger.error(f"❌ CRITICAL: {new_block} features missing!")
         raise ValueError("Feature generation failed")
         
    # Save
    out_path = f"{TEMP_DIR}/Q8_features.parquet"
    df_features.to_parquet(out_path)
    logger.info(f"Features saved to {out_path}")
    logger.info("Phase Q Step 1 (Iteration 8) Complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

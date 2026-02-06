
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
    logger.info("🚀 Starting Phase Q Step 1 (Iteration 3): Risk Stats Feature Generation")
    
    # Config
    START_DATE = "2015-01-01"
    END_DATE = "2024-12-31"
    
    # Feature Set (Base Q1 + Risk Stats)
    # Note: relative_stats in Q2 was rejected. We use Q1 set + risk_stats.
    # Wait, Q1 set used 'relative_stats' (z-score only) which is default.
    # Q2 modified 'relative_stats' to include Diff.
    # We should use the pipeline logic which determines what columns are outputted.
    # The pipeline code currently HAS the diff logic if we touched it in file system.
    # If we want to revert Q2 changes (remove Diff), we should modify feature_pipeline.py or just ignore them.
    # Since Q2 was rejected, we should ideally revert the code change in feature_pipeline.py to avoid noise,
    # OR we can just generate them and not use them in Config.
    # Generating them is fine.
    
    FEATURE_BLOCKS = [
        'base_attributes', 'history_stats', 'jockey_stats', 
        'pace_stats', 'bloodline_stats', 'training_stats',
        'burden_stats', 'changes_stats', 'aptitude_stats', 
        'speed_index_stats', 'pace_pressure_stats',
        'relative_stats', 'jockey_trainer_stats',
        'class_stats',
        'risk_stats' # New
    ]
    
    # We use data/features_q3 as cache dir
    CACHE_DIR = "data/features_q3"
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # 1. Copy Cache from Q1 (Base) to Q3
    # Q2 has modified relative_stats. Q1 has original.
    # We want Q1 base.
    # Copy data/features_q1/* to data/features_q3/
    logger.info("Copying Q1 cache to Q3...")
    
    # Checking Q1 cache
    q1_cache = "data/features_q1"
    if os.path.exists(q1_cache):
        import shutil
        # Copy file by file to avoid errors
        import glob
        files = glob.glob(f"{q1_cache}/*.parquet")
        for f in files:
            bn = os.path.basename(f)
            dst = os.path.join(CACHE_DIR, bn)
            if not os.path.exists(dst):
                 shutil.copy2(f, dst)
        logger.info("  -> Copied Q1 cache.")
    else:
        logger.warning("  -> Q1 cache not found! Starting fresh.")
        
    # 2. Data Loading (M5 or Q1 Temp)
    # We can reuse Q1 targets/raw temp file.
    TEMP_DIR = "data/temp_q3"
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    q1_target_path = "data/temp_q1/Q1_targets.parquet"
    q1_temp_raw_base = "data/temp_q1/year_" # year_{year}.parquet
    
    if not os.path.exists(q1_target_path):
        logger.error("Q1 Targets not found. Please run Q1 Step 1 first.")
        # Fallback to full load logic if needed, but assuming Q1 exists
        return

    # Load All Years
    logger.info("Loading Raw Data from Q1 Cache...")
    
    years = range(2015, 2025)
    raw_dfs = []
    for y in years:
        fpath = f"{q1_temp_raw_base}{y}.parquet"
        if os.path.exists(fpath):
            raw_dfs.append(pd.read_parquet(fpath))
        else:
            logger.warning(f"Missing year {y} in Q1 temp")
            
    if not raw_dfs:
        logger.error("No raw data found!")
        return
        
    clean_df = pd.concat(raw_dfs, ignore_index=True)
    logger.info(f"Loaded {len(clean_df)} rows.")
    
    # Save Q3 Targets (Same as Q1)
    target_cols = ['race_id', 'horse_number', 'rank', 'date']
    clean_df[target_cols].to_parquet(f"{TEMP_DIR}/Q3_targets.parquet")
    
    # 3. Feature Pipeline
    logger.info(f"Generating Features...")
    pipeline = FeaturePipeline(cache_dir=CACHE_DIR)
    
    # Force re-compute of new block if it somehow exists (unlikely)
    # risk_stats.parquet
    risk_cache = os.path.join(CACHE_DIR, "risk_stats.parquet")
    if os.path.exists(risk_cache):
        os.remove(risk_cache)
        
    df_features = pipeline.load_features(clean_df, FEATURE_BLOCKS)
    
    # Verify
    if 'rank_std_5' not in df_features.columns:
         logger.error("❌ CRITICAL: risk_stats features missing!")
         raise ValueError("Feature generation failed")
         
    # Save
    out_path = f"{TEMP_DIR}/Q3_features.parquet"
    df_features.to_parquet(out_path)
    logger.info(f"Features saved to {out_path}")
    logger.info("Phase Q Step 1 (Iteration 3) Complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

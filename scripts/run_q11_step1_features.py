
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
    logger.info("🚀 Starting Phase Q Step 1 (Iteration 11): Race Dynamics Feature Generation")
    
    # Feature Set (Base Q8 + Race Dynamics)
    # Note: Excluding Q9, Q10 features.
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
        'interval_aptitude',
        'race_dynamics' # New
    ]
    
    # We use data/features_q11 as cache dir
    CACHE_DIR = "data/features_q11"
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # 1. Copy Cache from Q8 to Q11
    # Q8 is the approved base.
    logger.info("Copying Q8 cache to Q11...")
    
    q8_cache = "data/features_q8"
    if os.path.exists(q8_cache):
        import shutil
        import glob
        files = glob.glob(f"{q8_cache}/*.parquet")
        for f in files:
            bn = os.path.basename(f)
            dst = os.path.join(CACHE_DIR, bn)
            if not os.path.exists(dst):
                 shutil.copy2(f, dst)
        logger.info("  -> Copied Q8 cache.")
    else:
        logger.warning("  -> Q8 cache not found! Starting fresh.")
        
    # 2. Data Loading
    TEMP_DIR = "data/temp_q11"
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    q1_target_path = "data/temp_q1/Q1_targets.parquet" 
    q1_temp_raw_base = "data/temp_q1/year_" 
    
    # Load All Years
    logger.info("Loading Raw Data...")
    
    years = range(2015, 2025)
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
    
    # Save Q11 Targets
    target_cols = ['race_id', 'horse_number', 'rank', 'date']
    if 'date' in clean_df.columns:
        clean_df[target_cols].to_parquet(f"{TEMP_DIR}/Q11_targets.parquet")
    else:
         logger.error("Date column missing in clean_df")
         
    # 3. Feature Pipeline
    logger.info(f"Generating Features...")
    pipeline = FeaturePipeline(cache_dir=CACHE_DIR)
    
    # Force re-compute of new block
    new_block = "race_dynamics"
    new_cache = os.path.join(CACHE_DIR, f"{new_block}.parquet")
    if os.path.exists(new_cache):
        os.remove(new_cache)
        
    df_features = pipeline.load_features(clean_df, FEATURE_BLOCKS)
    
    # Verify
    if 'front_runner_count' not in df_features.columns:
         logger.error(f"❌ CRITICAL: {new_block} features missing!")
         raise ValueError("Feature generation failed")
         
    # Save
    out_path = f"{TEMP_DIR}/Q11_features.parquet"
    df_features.to_parquet(out_path)
    logger.info(f"Features saved to {out_path}")
    logger.info("Phase Q Step 1 (Iteration 11) Complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

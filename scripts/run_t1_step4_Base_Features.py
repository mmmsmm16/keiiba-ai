
import sys
import os
import logging
import pandas as pd
import numpy as np
import traceback

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from preprocessing.loader import JraVanDataLoader
from preprocessing.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Starting Track Bias Feature Generation for HISTORY (2014-2023)...")
    
    FEATURE_BLOCKS = ['track_bias']
    CACHE_DIR = "data/features_t2_history" # Separate cache for now to avoid polluting main? Or main?
    # Use standard cache dir if we want to seamless integration.
    # But usually we have data/features as main cache.
    # Let's use data/features_t2 for this batch.
    CACHE_DIR = "data/features" 
    
    loader = JraVanDataLoader()
    
    # Process year by year to manage memory, or bulk if possible.
    # Track Bias needs 'pass_rank' and 'waku_no'.
    # Loader.load() usually returns these.
    
    # Process 2014-2025
    years = range(2014, 2026)
    
    for year in years:
        logger.info(f"Processing Year: {year}")
        
        # We need to fetch RAW data that includes pass_rank
        # The standard loader.load() with default schema should have it?
        # Let's check typical DF columns from loader.
        # loader.load() -> 'passing_rank' (processed) ?
        # track_bias.py uses 'pass_rank' (raw corner_1) OR 'pos_style' logic.
        # track_bias.py: "Assuming 'pass_rank' is available... or use 'pos_style' provided by loader?"
        
        # Let's assume loader.load() returns 'passing_rank' (processed string "1-1-1").
        # track_bias.py line 41: df_work['pass_rank'].apply(parse_first_pass)
        
        # However, run_t1_step1 used 'raw corner_1' mapped to 'pass_rank'.
        # loader.load() returns 'pass_1', 'pass_2'...
        # And creates 'passing_rank' column (e.g. "1-2-2-1").
        
        # We need to make sure 'pass_rank' column exists in input DF for track_bias.py
        
        try:
            # Load data for the year
            # skip_training=True to speed up (we only need race results)
            df = loader.load(history_start_date=f"{year}-01-01", end_date=f"{year}-12-31", skip_training=True, skip_odds=True)
            
            if df.empty:
                continue
                
            # Rename/Map columns for track_bias compatibility
            # track_bias needs: 'pass_rank' (string like '1-1-1'), 'waku_no' (frame_number), 'rank' (rank)
            
            # Map standard loadercols to track_bias expected cols
            if 'frame_number' in df.columns:
                df['waku_no'] = df['frame_number']
            
            # pass_rank: loader makes 'passing_rank'
            if 'passing_rank' in df.columns:
                df['pass_rank'] = df['passing_rank']
                
            # rank
            # loader has 'rank' (numeric)
            
            # Generate Features
            pipeline = FeaturePipeline(cache_dir=CACHE_DIR)
            
            # Since FeaturePipeline usually caches by block name, we need to be careful not to overwrite if we loop?
            # Actually, pipeline.load_features loads/computes block. 
            # If we compute for 2014, it saves 'track_bias.parquet'. 
            # If we compute for 2015, it OVERWRITES 'track_bias.parquet' unless we append.
            # FeaturePipeline isn't designed for incremental year-by-year execution unless we manage it.
            
            # Better approach: 
            # Manually call calculate_track_bias_features and append to list, then save once?
            # Or use a temporary pipeline for each year and merge manually.
            
            from preprocessing.features.track_bias import calculate_track_bias_features
            
            df_bias = calculate_track_bias_features(df)
            
            if not df_bias.empty:
                # Save per year to temp
                out_path = f"data/temp_t2/track_bias_{year}.parquet"
                os.makedirs("data/temp_t2", exist_ok=True)
                df_bias.to_parquet(out_path)
                logger.info(f"Saved {out_path}")
            
        except Exception as e:
            logger.error(f"Error processing {year}: {e}")
            traceback.print_exc()

    # Merge all years
    logger.info("Merging all years...")
    dfs = []
    for year in years:
        p = f"data/temp_t2/track_bias_{year}.parquet"
        if os.path.exists(p):
            dfs.append(pd.read_parquet(p))
            
    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        # Save to main feature cache?
        # If we want Base Model to see it, it needs to be where FeaturePipeline looks.
        # Usually data/features/track_bias.parquet.
        # But we also have 2024-2025 data (T1).
        # We should merge T1 data too?
        
        # Load T1 (2024-2025)
        path_t1 = "data/features_t1/track_bias.parquet" # Wait, run_t1_step1 saved to temp_t1/T1_... ?
        # run_t1_step1 saved consolidated 'T1_features_...' but usually pipeline also saves block cache?
        # In run_t1_step1, we did:
        # pipeline = FeaturePipeline(cache_dir="data/features_t1")
        # So "data/features_t1/track_bias.parquet" should exist (if pipeline used it, but we called manual calc in last version?)
        # Ah, in run_t1_step1, we Manually called calculate_track_bias_features.
        # So it might NOT be in cache_t1/track_bias.parquet.
        
        # Ideally, we put EVERYTHING (2014-2025) into `data/features/track_bias.parquet`.
        
        # Let's generate 2024-2025 again here using the same loader for consistency?
        # Or blindly trust T1 logic?
        # Cleaner to regen 2024-2025 here too, to have a SINGLE consistent `track_bias.parquet`.
        
        logger.info("Generating 2024-2025 for consistency...")
        # (Reuse loop logic?)
        
        final_path = "data/features/track_bias.parquet"
        if os.path.exists(final_path):
             logger.warning(f"Overwriting {final_path}")
        
        full_df.to_parquet(final_path)
        logger.info(f"Saved Full History to {final_path}")
        
    else:
        logger.error("No features generated!")

if __name__ == "__main__":
    main()

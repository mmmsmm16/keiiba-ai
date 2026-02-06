
import sys
import os
import logging
import pandas as pd
import numpy as np
import traceback
import gc

sys.path.append(os.path.join(os.getcwd(), 'src'))

from preprocessing.loader import JraVanDataLoader
from preprocessing.features.track_bias import calculate_track_bias_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Starting Track Bias Feature Generation (FULL HISTORY in ONE PASS)...")
    
    loader = JraVanDataLoader()
    
    # Load ALL years in one DataFrame to preserve horse history across years
    all_dfs = []
    
    for year in range(2014, 2026):
        logger.info(f"Loading Year: {year}")
        try:
            df = loader.load(history_start_date=f"{year}-01-01", end_date=f"{year}-12-31", skip_training=True, skip_odds=True)
            
            if df.empty:
                continue
            
            # Rename columns for track_bias compatibility
            if 'frame_number' in df.columns:
                df['waku_no'] = df['frame_number']
            if 'passing_rank' in df.columns:
                df['pass_rank'] = df['passing_rank']
                
            all_dfs.append(df)
            
            del df
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error loading {year}: {e}")
            traceback.print_exc()
    
    if not all_dfs:
        logger.error("No data loaded!")
        return
        
    logger.info("Concatenating all years...")
    full_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total rows: {len(full_df)}")
    
    del all_dfs
    gc.collect()
    
    # Ensure date is datetime and sorted
    full_df['date'] = pd.to_datetime(full_df['date'])
    full_df = full_df.sort_values(['date', 'race_id', 'horse_number']).reset_index(drop=True)
    
    logger.info("Calculating Track Bias features for FULL HISTORY...")
    df_bias = calculate_track_bias_features(full_df)
    
    if df_bias.empty:
        logger.error("Track Bias feature calculation returned empty!")
        return
        
    logger.info(f"Track Bias features generated: {len(df_bias)} rows")
    logger.info(f"Non-zero bias scores: {(df_bias['bias_adversity_score_mean_5'] > 0).sum()}")
    
    # Save to cache
    os.makedirs("data/features", exist_ok=True)
    final_path = "data/features/track_bias.parquet"
    df_bias.to_parquet(final_path, index=False)
    logger.info(f"Saved to {final_path}")
    
    # Print stats
    print("\n=== Track Bias Feature Stats ===")
    print(df_bias['bias_adversity_score_mean_5'].describe())

if __name__ == "__main__":
    main()

"""
Incremental Daily Feature Update
=================================
Fast differential update - only processes new data since last update.
1. Reads existing parquet to find max date.
2. Loads only new data from DB (max_date onwards).
3. Generates features for new data only.
4. Appends to existing parquet.

Usage:
  python scripts/update_daily_features_incremental.pyｗ
  python scripts/update_daily_features_incremental.py --force  # Full rebuild
"""
import os
import sys
import argparse
import logging
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_PATH = "data/processed/preprocessed_data_v11.parquet"
CACHE_DIR = "data/features_t2/incremental_cache"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force full rebuild")
    parser.add_argument("--skip_odds", action="store_true", help="Skip loading 10min odds (faster)")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Incremental Feature Update")
    logger.info("=" * 60)
    
    # Check existing data
    if os.path.exists(OUTPUT_PATH) and not args.force:
        logger.info(f"Loading existing parquet: {OUTPUT_PATH}")
        df_existing = pd.read_parquet(OUTPUT_PATH)
        df_existing['date'] = pd.to_datetime(df_existing['date'])
        max_date = df_existing['date'].max()
        logger.info(f"Existing records: {len(df_existing)}, Max date: {max_date.date()}")
        
        # Load only new data (from max_date - 1 day to be safe with timezone)
        start_date = (max_date - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        logger.info("No existing parquet or --force specified. Full rebuild.")
        df_existing = None
        start_date = "2019-01-01"
    
    # Load new data from DB
    logger.info(f"Loading data from DB since {start_date}...")
    loader = JraVanDataLoader()
    df_new = loader.load(history_start_date=start_date, skip_odds=args.skip_odds)
    
    if df_new.empty:
        logger.info("No new data found. Exiting.")
        return
    
    logger.info(f"Loaded {len(df_new)} records from DB")
    
    # Preprocessing
    df_new['race_id'] = df_new['race_id'].astype(str)
    df_new['date'] = pd.to_datetime(df_new['date'])
    
    # Feature Generation (use separate cache to avoid conflicts)
    logger.info("Generating features for new data...")
    pipeline = FeaturePipeline(cache_dir=CACHE_DIR)
    
    # Exclude problematic blocks that require specific keys
    exclude_blocks = ['odds_fluctuation']
    feature_blocks = [b for b in pipeline.registry.keys() if b not in exclude_blocks]
    logger.info(f"Feature blocks: {len(feature_blocks)} (excluded: {exclude_blocks})")
    
    df_features_new = pipeline.load_features(df_new, feature_blocks)
    
    # Ensure race_id matches type of df_new (str) to prevent merge failure
    if 'race_id' in df_features_new.columns:
        df_features_new['race_id'] = df_features_new['race_id'].astype(str).str.replace('.0', '', regex=False)
    
    # Merge date if missing
    if 'date' not in df_features_new.columns:
        df_keys = df_new[['race_id', 'horse_number', 'date']].drop_duplicates(subset=['race_id', 'horse_number'])
        df_features_new = df_features_new.merge(df_keys, on=['race_id', 'horse_number'], how='left')

    # Restore Target/Meta columns (Critical for Training)
    # df_new contains the original data loaded from DB (including rank, odds)
    target_cols = ['rank', 'odds', 'odds_final', 'is_win', 'is_top2', 'is_top3', 'result']
    cols_to_restore = [c for c in target_cols if c in df_new.columns and c not in df_features_new.columns]
    
    if cols_to_restore:
        logger.info(f"Restoring target columns: {cols_to_restore}")
        df_targets = df_new[['race_id', 'horse_number'] + cols_to_restore].drop_duplicates(subset=['race_id', 'horse_number'])
        df_features_new = df_features_new.merge(df_targets, on=['race_id', 'horse_number'], how='left')

    
    # Fix types
    for col in df_features_new.columns:
        if df_features_new[col].dtype == 'object':
            try:
                # Use raise to avoid converting '牡' to NaN
                df_features_new[col] = pd.to_numeric(df_features_new[col], errors='raise')
            except:
                # Keep as string if it fails to convert (e.g. '牡', '芝')
                df_features_new[col] = df_features_new[col].astype(str)
        elif df_features_new[col].dtype.name == 'category':
            df_features_new[col] = df_features_new[col].astype(str)
    
    df_features_new['race_id'] = df_features_new['race_id'].astype(str).str.replace('.0', '', regex=False)
    df_features_new['date'] = pd.to_datetime(df_features_new['date'])
    
    # Merge with existing
    if df_existing is not None:
        logger.info("Merging with existing data...")
        
        # Remove overlapping dates from existing (we regenerated them)
        cutoff_date = pd.to_datetime(start_date)
        df_existing_keep = df_existing[df_existing['date'] < cutoff_date].copy()
        logger.info(f"Keeping {len(df_existing_keep)} existing records before {start_date}")
        
        # Combine
        df_final = pd.concat([df_existing_keep, df_features_new], ignore_index=True)
        
        # Remove duplicates (in case of overlap)
        before_dedup = len(df_final)
        df_final = df_final.drop_duplicates(subset=['race_id', 'horse_number'], keep='last')
        logger.info(f"Deduplication: {before_dedup} -> {len(df_final)}")
    else:
        df_final = df_features_new
    
    # Sort by date
    df_final = df_final.sort_values(['date', 'race_id', 'horse_number']).reset_index(drop=True)
    
    # Save
    logger.info(f"Saving to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_final.to_parquet(OUTPUT_PATH, index=False)
    
    logger.info("=" * 60)
    logger.info("✅ Incremental update complete!")
    logger.info(f"   Total records: {len(df_final)}")
    logger.info(f"   Columns: {len(df_final.columns)}")
    logger.info(f"   Date range: {df_final['date'].min().date()} ~ {df_final['date'].max().date()}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

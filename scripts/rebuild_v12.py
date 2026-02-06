
"""
Rebuild preprocessed_data_v12.parquet (Batch 1: Mining & Track Aptitude)
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append('/workspace')

from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_PATH = "data/processed/preprocessed_data_v12.parquet"
CACHE_DIR = "data/features_t2/rebuild_v12"

def main():
    logger.info("=" * 60)
    logger.info("Rebuilding preprocessed_data_v12.parquet (Batch 1)")
    logger.info("=" * 60)
    
    # Load all data from DB
    logger.info("Step 1: Loading data from DB...")
    loader = JraVanDataLoader()
    # Loading enough history for stable validation
    # Skip time-series odds as requested to speed up (mainly used for EV)
    df = loader.load(history_start_date="2019-01-01", skip_odds=True)
    
    logger.info(f"Loaded {len(df)} records")
    
    df['race_id'] = df['race_id'].astype(str)
    df['date'] = pd.to_datetime(df['date'])
    
    # Use FeaturePipeline to generate features
    logger.info("Step 2: Generating features...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    pipeline = FeaturePipeline(cache_dir=CACHE_DIR)
    
    # Use all registered blocks
    feature_blocks = list(pipeline.registry.keys())
    
    logger.info(f"Feature blocks to generate ({len(feature_blocks)}): {feature_blocks}")
    
    df_features = pipeline.load_features(df, feature_blocks)
    
    logger.info(f"Generated features: {len(df_features)} records, {len(df_features.columns)} columns")
    
    # Merge target columns and date back
    target_cols = ['rank', 'odds', 'popularity', 'time', 'date']
    cols_to_add = [c for c in target_cols if c not in df_features.columns and c in df.columns]
    
    if cols_to_add:
        logger.info(f"Merging target columns back: {cols_to_add}")
        df_keys = df[['race_id', 'horse_number'] + cols_to_add].drop_duplicates(subset=['race_id', 'horse_number'])
        df_features = df_features.merge(df_keys, on=['race_id', 'horse_number'], how='left')
    
    # Fix race_id format
    df_features['race_id'] = df_features['race_id'].astype(str).str.replace('.0', '', regex=False)
    
    # Save
    logger.info(f"Step 4: Saving to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_features.to_parquet(OUTPUT_PATH, index=False)
    
    logger.info("=" * 60)
    logger.info(f"✅ Rebuild complete!")
    logger.info(f"   Records: {len(df_features)}")
    logger.info(f"   Columns: {len(df_features.columns)}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

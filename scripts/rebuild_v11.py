"""
Rebuild preprocessed_data_v11.parquet from scratch
Uses FeaturePipeline to generate all features
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

OUTPUT_PATH = "data/processed/preprocessed_data_v11.parquet"
CACHE_DIR = "data/features_t2/rebuild_v11"
CONFIG_PATH = "models/experiments/exp_t2_refined_v3/config.yaml"


def main():
    logger.info("=" * 60)
    logger.info("Rebuilding preprocessed_data_v11.parquet")
    logger.info("=" * 60)
    
    # Load all data from DB
    logger.info("Step 1: Loading data from DB...")
    loader = JraVanDataLoader()
    df = loader.load(history_start_date="2019-01-01")
    
    logger.info(f"Loaded {len(df)} records")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    df['race_id'] = df['race_id'].astype(str)
    df['date'] = pd.to_datetime(df['date'])
    
    # Use FeaturePipeline to generate features
    logger.info("Step 2: Generating features...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Use FeaturePipeline to generate features
    logger.info("Step 2: Generating features...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    pipeline = FeaturePipeline(cache_dir=CACHE_DIR)
    
    # Use all registered blocks
    feature_blocks = list(pipeline.registry.keys())
    # Exclude experimental or heavy blocks if known (optional, but let's include all for safety)
    # feature_blocks = [b for b in feature_blocks if b not in ['deep_lag_extended']] # Example exclusion
    
    logger.info(f"Feature blocks to generate ({len(feature_blocks)}): {feature_blocks}")
    
    df_features = pipeline.load_features(df, feature_blocks)
    
    logger.info(f"Generated features: {len(df_features)} records, {len(df_features.columns)} columns")
    
    # Add date column back if missing
    if 'date' not in df_features.columns:
        logger.info("Merging date column back...")
        df_keys = df[['race_id', 'horse_number', 'date']].drop_duplicates(subset=['race_id', 'horse_number'])
        df_features = df_features.merge(df_keys, on=['race_id', 'horse_number'], how='left')
    
    # Fix column types for parquet
    logger.info("Step 3: Fixing column types...")
    for col in df_features.columns:
        if df_features[col].dtype == 'object':
            try:
                df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
            except:
                df_features[col] = df_features[col].astype(str)
        elif df_features[col].dtype.name == 'category':
            df_features[col] = df_features[col].astype(str)
    
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
    logger.info(f"   Date range: {df_features['date'].min()} to {df_features['date'].max()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

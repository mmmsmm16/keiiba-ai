"""
Safe Update Daily Features
==========================
Uses the robust rebuild_v11 logic to update preprocessed_data_v11.parquet.
To ensure consistency and avoid corruption, this script:
1. Clears the feature cache (forcing a full recompute).
2. Loads all data from DB.
3. Generates all features using FeaturePipeline.
4. Overwrites preprocessed_data_v11.parquet.
"""
import os
import sys
import shutil
import logging
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_PATH = "data/processed/preprocessed_data_v11.parquet"
CACHE_DIR = "data/features_t2/rebuild_v11"

def main():
    logger.info("=" * 60)
    logger.info("Safe Update: Rebuilding preprocessed_data_v11.parquet")
    logger.info("=" * 60)

    # Safety: Clear cache to ensure we pick up new data and don't mix old/new alignment
    if os.path.exists(CACHE_DIR):
        logger.warning(f"Removing cache directory to force rebuild: {CACHE_DIR}")
        shutil.rmtree(CACHE_DIR)
    
    # Load all data
    logger.info("Step 1: Loading data from DB...")
    loader = JraVanDataLoader()
    # Load from 2019 to match v11 definition
    df = loader.load(history_start_date="2019-01-01")
    
    logger.info(f"Loaded {len(df)} records")
    
    # Preprocessing
    df['race_id'] = df['race_id'].astype(str)
    df['date'] = pd.to_datetime(df['date'])
    
    # Feature Generation
    logger.info("Step 2: Generating features...")
    pipeline = FeaturePipeline(cache_dir=CACHE_DIR)
    
    # Use all registered blocks
    feature_blocks = list(pipeline.registry.keys())
    logger.info(f"Feature blocks: {len(feature_blocks)}")
    
    df_features = pipeline.load_features(df, feature_blocks)
    
    # Date merge if needed
    if 'date' not in df_features.columns:
        logger.info("Merging date column back...")
        df_keys = df[['race_id', 'horse_number', 'date']].drop_duplicates(subset=['race_id', 'horse_number'])
        df_features = df_features.merge(df_keys, on=['race_id', 'horse_number'], how='left')
        
    # Fix types like in rebuild_v11
    logger.info("Step 3: Fixing column types...")
    for col in df_features.columns:
        if df_features[col].dtype == 'object':
            try:
                df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
            except:
                df_features[col] = df_features[col].astype(str)
        elif df_features[col].dtype.name == 'category':
            df_features[col] = df_features[col].astype(str)

    # Fix race_id
    df_features['race_id'] = df_features['race_id'].astype(str).str.replace('.0', '', regex=False)
            
    # Save
    logger.info(f"Step 4: Saving to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_features.to_parquet(OUTPUT_PATH, index=False)
    
    logger.info("=" * 60)
    logger.info("✅ Update complete!")
    logger.info(f"   Records: {len(df_features)}")
    logger.info(f"   Columns: {len(df_features.columns)}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

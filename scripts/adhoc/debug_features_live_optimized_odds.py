
import sys
import os
import pandas as pd
import numpy as np
import logging

# Ensure root is in path
sys.path.append('/workspace')
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebugFeatures")

def main():
    date_str = "20260124"
    target_date = "2026-01-24"
    
    loader = JraVanDataLoader()
    # Load raw
    df_raw = loader.load(history_start_date="2016-01-01", end_date=target_date, skip_odds=False)
    
    logger.info(f"Raw Data Shape: {df_raw.shape}")
    logger.info(f"Raw Columns: {df_raw.columns.tolist()[:10]}...")
    
    # Check ID types
    logger.info(f"Race ID Type: {df_raw['race_id'].dtype}")
    logger.info(f"Horse ID Type: {df_raw['horse_id'].dtype if 'horse_id' in df_raw.columns else 'MISSING'}")
    
    # Fix types as in production script
    if 'race_id' in df_raw.columns:
        df_raw['race_id'] = df_raw['race_id'].astype(str)
    if 'horse_number' in df_raw.columns:
        df_raw['horse_number'] = pd.to_numeric(df_raw['horse_number'], errors='coerce').fillna(0).astype(int)
        
    # Sample row
    logger.info("Sample Raw Row:")
    print(df_raw.iloc[0][['race_id', 'horse_number', 'horse_id', 'jockey_id']].to_dict())

    # Load Model Feature Lists
    try:
        v13_feats = pd.read_csv('models/experiments/exp_lambdarank_hard_weighted/features.csv', header=None)[0].tolist()
        # Handle cases where header exists
        if v13_feats[0] == 'feature': v13_feats = v13_feats[1:]
        
        v14_feats = pd.read_csv('models/experiments/exp_gap_v14_production/features.csv')['feature'].tolist()
    except Exception as e:
        logger.error(f"Failed to load feature lists: {e}")
        return

    logger.info(f"V13 Features Count: {len(v13_feats)}")
    logger.info(f"V14 Features Count: {len(v14_feats)}")

    # Run Pipeline
    pipeline = FeaturePipeline(cache_dir="data/features_v14/prod_cache")
    # Use ALL registered blocks automatically
    blocks = list(pipeline.registry.keys())
    logger.info(f"Loaded {len(blocks)} blocks from registry.")
    
    # Check if raw data has rows
    if df_raw.empty:
        logger.error("Raw df is empty!")
        return

    # Simulate Prediction Prep (Merging)
    # Note: Some blocks might be redundant or fail if dependencies missing, but registry should handle it
    df_features = pipeline.load_features(df_raw, blocks, force=True)
    logger.info(f"Generated Features Shape: {df_features.shape}")
    
    # Merge
    key_cols = ['race_id', 'horse_number']
    # Ensure raw keys match feature keys types (features usually str/int)
    df_raw['race_id'] = df_raw['race_id'].astype(str)
    df_raw['horse_number'] = df_raw['horse_number'].fillna(0).astype(int)
    
    # Feature df usually has race_id as object/string
    df_features['race_id'] = df_features['race_id'].astype(str)
    df_features['horse_number'] = df_features['horse_number'].fillna(0).astype(int)

    df_merged = pd.merge(df_raw[key_cols + ['date', 'odds']], df_features, on=key_cols, how='left', suffixes=('', '_feat'))
    
    # Fix date if collision happened
    if 'date' not in df_merged.columns and 'date_' in df_merged.columns: # unlikely with empty suffix
         pass
         
    logger.info(f"Merged Columns: {df_merged.columns.tolist()[:10]}...")

    # Calculate Derived (Simulate script)
    # odds_10min alias
    if 'odds_10min' not in df_merged.columns:
        df_merged['odds_10min'] = df_merged['odds'].fillna(10.0)
    
    # odds_rank_10min
    df_merged['odds_rank_10min'] = df_merged.groupby('race_id')['odds_10min'].rank(method='min')
    
    # odds_final column
    if 'odds_final' not in df_merged.columns:
        df_merged['odds_final'] = df_merged['odds']

    # Filter Target Date
    if 'date' not in df_merged.columns:
        logger.error(f"Date column missing! Cols: {df_merged.columns}")
        return

    df_target = df_merged[df_merged['date'] == target_date].copy()
    logger.info(f"Target Date Rows: {len(df_target)}")
    
    if df_target.empty:
        logger.error("No rows for target date!")
        return

    # Validation Function
    def validate_features(name, feats, df):
        logger.info(f"\n--- Validating {name} Features ---")
        missing = [c for c in feats if c not in df.columns]
        if missing:
            logger.error(f"MISSING columns ({len(missing)}): {missing[:5]}...")
        else:
            logger.info("All columns present.")
            
        # Check Zero/NaN
        valid_cols = [c for c in feats if c in df.columns]
        df_sub = df[valid_cols]
        
        nans = df_sub.isna().sum()
        zeros = (df_sub == 0).sum()
        
        # Report heavy NaN/Zero columns
        for c in valid_cols:
            n_nan = nans[c]
            n_zero = zeros[c]
            # Zero is weird for some, OK for others. NaN is bad.
            if n_nan > 0:
                logger.warning(f"  {c}: {n_nan} NaNs")
            if n_zero == len(df_sub):
                logger.warning(f"  {c}: ALL ZEROS warning")
                
        # Specific Check
        if 'odds_rank_10min' in valid_cols:
            logger.info(f"  odds_rank_10min stats: {df_sub['odds_rank_10min'].describe().to_dict()}")
        if 'sire_apt_rot' in valid_cols:
            logger.info(f"  sire_apt_rot stats: {df_sub['sire_apt_rot'].describe().to_dict()}")

    validate_features("V13", v13_feats, df_target)
    validate_features("V14", v14_feats, df_target)

if __name__ == "__main__":
    main()

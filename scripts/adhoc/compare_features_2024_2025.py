import pandas as pd
import glob
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = "data/features"
OUTPUT_FILE = "reports/feature_consistency_2024_2025.csv"

def main():
    logger.info("Starting Feature Consistency Check (2024 vs 2025)...")
    
    # 1. Load Base Attributes to establish Year/Index
    base_file = os.path.join(CACHE_DIR, "base_attributes.parquet")
    if not os.path.exists(base_file):
        logger.error(f"Base attributes not found at {base_file}")
        return

    logger.info(f"Loading Index from {base_file}...")
    df_base = pd.read_parquet(base_file)
    
    # Ensure race_id matches 2024/2025 pattern
    # The cache contains 2014-2025 data. We only care about 2024 and 2025.
    df_base['year'] = df_base['race_id'].astype(str).str[:4].astype(int)
    
    # Create masks
    mask_24 = df_base['year'] == 2024
    mask_25 = df_base['year'] == 2025
    
    logger.info(f"2024 Records: {mask_24.sum()}")
    logger.info(f"2025 Records: {mask_25.sum()}")
    
    if mask_25.sum() == 0:
        logger.error("No 2025 data found in cache! Did generation fail?")
        return

    # Get indices (assuming index is aligned if we load identically, but safer to join on index if present)
    # FeaturePipeline typically saves with RangeIndex. We rely on filename or merge keys.
    # Feature blocks usually have NO keys if managed by pipeline (just concatenated). 
    # WAIT: FeaturePipeline in this project usually saves blocks *with* keys or relies on exact row alignment?
    # Checking FeaturePipeline code... usually it relies on list order matching df_raw.
    # If we load parquet files independently, they should have the same index as df_raw IF they were generated in one batch.
    # Since we generated 2014-2025 in one go, row N in base_attributes corresponds to row N in any other block.
    # So using boolean mask on rows is valid.
    
    feature_files = glob.glob(os.path.join(CACHE_DIR, "*.parquet"))
    results = []

    for fpath in feature_files:
        fname = os.path.basename(fpath)
        if fname == "temp_merge_current.parquet": continue
        
        logger.info(f"Analyzing {fname}...")
        try:
            df_block = pd.read_parquet(fpath)
            
            # Validation: Row count must match base
            if len(df_block) != len(df_base):
                logger.warning(f"Skipping {fname}: Row count mismatch ({len(df_block)} vs {len(df_base)})")
                continue
                
            # Iterate columns
            for col in df_block.columns:
                if col in ['race_id', 'horse_number', 'date', 'year']: continue # Skip keys
                
                # Check 2024
                series_24 = df_block.loc[mask_24, col]
                series_25 = df_block.loc[mask_25, col]
                
                # Calculate Null Rate
                null_24 = series_24.isnull().mean()
                null_25 = series_25.isnull().mean()
                
                # Calculate Mean (if numeric)
                mean_24 = 0.0
                mean_25 = 0.0
                std_24 = 0.0
                std_25 = 0.0
                is_numeric = False
                
                if pd.api.types.is_numeric_dtype(series_24):
                    is_numeric = True
                    mean_24 = series_24.mean()
                    mean_25 = series_25.mean()
                    std_24 = series_24.std()
                    std_25 = series_25.std()
                
                # Deviation Score
                null_diff = abs(null_24 - null_25)
                # Z-score-like difference: (mean25 - mean24) / std24
                mean_drift = 0.0
                if is_numeric and std_24 > 0:
                    mean_drift = abs(mean_25 - mean_24) / std_24
                
                results.append({
                    'file': fname,
                    'feature': col,
                    'null_24': null_24,
                    'null_25': null_25,
                    'null_diff': null_diff,
                    'mean_24': mean_24,
                    'mean_25': mean_25,
                    'std_24': std_24,
                    'std_25': std_25,
                    'drift_score': mean_drift,
                    'is_numeric': is_numeric
                })
                
        except Exception as e:
            logger.error(f"Error processing {fname}: {e}")

    # Save Results
    df_res = pd.DataFrame(results)
    df_res.sort_values('drift_score', ascending=False, inplace=True)
    df_res.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved consistency report to {OUTPUT_FILE}")
    
    # Print Top Drift Features
    logger.info("=== Top 10 Features by Mean Drift ===")
    print(df_res[df_res['is_numeric']].head(10)[['feature', 'mean_24', 'mean_25', 'drift_score']])

    # Print Top Null Diff Features
    logger.info("=== Top 10 Features by Null Rate Diff ===")
    df_res.sort_values('null_diff', ascending=False, inplace=True)
    print(df_res.head(10)[['feature', 'null_24', 'null_25', 'null_diff']])

if __name__ == "__main__":
    main()

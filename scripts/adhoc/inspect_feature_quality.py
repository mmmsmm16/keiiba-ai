
import pandas as pd
import glob
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Inspecting 2025 Feature Quality...")
    
    # Target blocks that heavily rely on history
    target_blocks = ['deep_lag_extended', 'form_trend', 'relative_expansion']
    
    for block in target_blocks:
        path = f"data/features/{block}.parquet"
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            continue
            
        logger.info(f"Checking {block}...")
        df = pd.read_parquet(path)
        
        # Filter for 2025 (assume race_id starts with '2025')
        df['race_id'] = df['race_id'].astype(str)
        df_2025 = df[df['race_id'].str.startswith('2025')]
        
        if df_2025.empty:
            logger.warning(f"  No 2025 data in {block}")
            continue
            
        logger.info(f"  2025 Rows: {len(df_2025)}")
        
        # Check zeros and nulls in feature columns (skip IDs)
        feat_cols = [c for c in df_2025.columns if c not in ['race_id', 'horse_number']]
        
        for col in feat_cols[:5]: # Check first 5 cols as sample
            n_zeros = (df_2025[col] == 0).sum()
            n_nulls = df_2025[col].isnull().sum()
            rate_zero = n_zeros / len(df_2025)
            rate_null = n_nulls / len(df_2025)
            
            logger.info(f"    {col}: Zero={rate_zero:.2%}, Null={rate_null:.2%} (Mean={df_2025[col].mean():.4f})")
            
if __name__ == "__main__":
    main()

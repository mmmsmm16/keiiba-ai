
import sys
import os
import pandas as pd
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.preprocessing.feature_pipeline import FeaturePipeline

def check_lag_features():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create dummy data for one horse
    data = {
        'horse_id': ['H001'] * 10,
        'date': pd.date_range(start='2023-01-01', periods=10, freq='M'),
        'rank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'time_diff': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'race_id': [f'R{i}' for i in range(10)],
        'horse_number': [1] * 10,
        'honshokin': [1000] * 10,
        'fukashokin': [0] * 10,
        'grade_code': ['G1'] * 10
    }
    df = pd.DataFrame(data)

    pipeline = FeaturePipeline(cache_dir="tmp/features_test")
    
    # Compute history stats (force=True to skip cache)
    df_feats = pipeline._compute_history_stats(df)
    
    logger.info("Generated Columns:")
    logger.info(df_feats.columns.tolist())
    
    # Check for lag1 to lag5
    for i in range(1, 6):
        col = f"lag{i}_rank"
        if col in df_feats.columns:
            logger.info(f"OK: {col} is present.")
            # Verify value for the last row (index 9)
            # lag1 should be rank at index 8 (9)
            # lag5 should be rank at index 4 (5)
            val = df_feats.iloc[9][col]
            logger.info(f"  Value at last row for {col}: {val}")
        else:
            logger.error(f"FAIL: {col} is missing!")

    return df_feats

if __name__ == "__main__":
    check_lag_features()

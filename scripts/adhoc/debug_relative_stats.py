
import os
import sys
import pandas as pd
import numpy as np
import logging

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.preprocessing.feature_pipeline import FeaturePipeline
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.cleansing import DataCleanser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_relative():
    logger.info("Loading data sample...")
    loader = JraVanDataLoader()
    # Load small subset? 
    # Or load full but just Limit?
    # Loader doesn't support limit easily but we can slice after load.
    
    # Load 2024 data only to be fast?
    # Or 2020-2024.
    df = loader.load(history_start_date='2023-01-01', end_date='2023-06-01', jra_only=True)
    logger.info(f"Loaded {len(df)} records. Columns: {df.columns.tolist()}")
    
    # Cleansing
    cleanser = DataCleanser()
    df = cleanser.cleanse(df)
    logger.info("Cleansed.")
    
    # Ensure date is datetime
    # df['date'] = pd.to_datetime(df['date']) # Cleanser might do this
    
    # Run pipeline block
    pipeline = FeaturePipeline(cache_dir="tmp/debug_cache")
    
    logger.info("Running pipeline.load_features...")
    # Simulate experiment list
    feature_blocks = [
        'base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 
        'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 
        'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 
        'relative_stats'
    ]
    res = pipeline.load_features(df, feature_blocks)
    logger.info("Success!")
    print(res.head())

if __name__ == "__main__":
    debug_relative()

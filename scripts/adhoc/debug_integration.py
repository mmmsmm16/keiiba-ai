import pandas as pd
import sys
import os
import shutil

# Adjust path to find src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline

def main():
    print("Loading small sample...")
    loader = JraVanDataLoader()
    # Load 1 year
    df = loader.load(history_start_date='2024-01-01', skip_odds=True, skip_training=False)
    df = df.head(1000)
    print(f"Loaded {len(df)} rows.")
    
    # Use temporary cache dir
    cache_dir = "data/features_t2/incremental_cache_debug_integ"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir)
    
    pipeline = FeaturePipeline(cache_dir=cache_dir)
    
    # 1. Test Pace Stats
    print("\n--- Testing Pace Stats Block ---")
    try:
        df_pace = pipeline.load_features(df, ['pace_stats'])
        if 'last_nige_rate' in df_pace.columns:
            print(f"last_nige_rate describe:\n{df_pace['last_nige_rate'].describe()}")
            # Print non-zero count
            nz = (df_pace['last_nige_rate'] > 0).sum()
            print(f"last_nige_rate > 0 count: {nz}")
        else:
            print("last_nige_rate NOT FOUND in output")
    except Exception as e:
        print(f"Pace Stats Failed: {e}")
        
    # 2. Test Bloodline Stats Block
    print("\n--- Testing Bloodline Stats Block ---")
    try:
        df_blood = pipeline.load_features(df, ['bloodline_stats'])
        if 'sire_heavy_win_rate' in df_blood.columns:
            print(f"sire_heavy_win_rate describe:\n{df_blood['sire_heavy_win_rate'].describe()}")
             # Print non-zero count
            nz = (df_blood['sire_heavy_win_rate'] > 0).sum()
            print(f"sire_heavy_win_rate > 0 count: {nz}")
        else:
             print("sire_heavy_win_rate NOT FOUND in output")
    except Exception as e:
        print(f"Bloodline Stats Failed: {e}")

if __name__ == "__main__":
    main()

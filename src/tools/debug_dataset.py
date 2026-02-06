import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.feature_pipeline import FeaturePipeline
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.cleansing import DataCleanser
from src.preprocessing.dataset import DatasetSplitter

def debug():
    print("=== Dataset Debug ===")
    
    # 1. Load Data
    print("Loading Data (2024 only to match valid set)...")
    loader = JraVanDataLoader()
    raw_df = loader.load(history_start_date='2024-01-01', end_date='2024-12-31', jra_only=True)
    cleanser = DataCleanser()
    clean_df = cleanser.cleanse(raw_df)
    
    # 2. Features
    print("Loading Features...")
    pipeline = FeaturePipeline(cache_dir="data/features")
    feature_blocks = ['base_attributes', 'history_stats']
    df = pipeline.load_features(clean_df, feature_blocks)
    
    # 3. Add Keys for Splitter
    # rank is needed for target creation if target not in df
    key_cols = ['race_id', 'date', 'horse_id', 'rank']
    for k in key_cols:
        if k not in df.columns and k in clean_df.columns:
            df[k] = clean_df[k]
    
    if 'year' not in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year

    # 4. Split
    print("Splitting usage DatasetSplitter...")
    splitter = DatasetSplitter()
    # Use 2024 as valid year so we get 'valid' set populated
    datasets = splitter.split_and_create_dataset(df, valid_year=2024)
    
    valid_set = datasets['valid']
    X = valid_set['X']
    y = valid_set['y']
    group = valid_set['group']
    
    print(f"Valid X Shape: {X.shape}")
    print(f"Valid y Shape: {y.shape}")
    print(f"Valid Group Shape: {group.shape}")
    print(f"Group Sum: {group.sum()}")
    
    # Validation 1: Group Sum
    if group.sum() != len(X):
        print(f"❌ MISMATCH: Group Sum {group.sum()} != len(X) {len(X)}")
    else:
        print("✅ Group Sum matches len(X)")
        
    # Validation 2: Check Sorting (Implicit check)
    # We can't check race_id in X because it's dropped.
    # But we can check if y groups match group sizes?
    # No, y is just targets.
    
    # Let's verify 'dataset.py' logic by inspecting a small sample manually using the logic
    # Re-run similar logic here
    print("Verifying Sort Logic...")
    
    df_valid = df[df['year'] == 2024].copy()
    df_sorted = df_valid.sort_values(['date', 'race_id'])
    
    # Check if df_sorted index matches X index? 
    # DatasetSplitter creates new X, reset index?
    # X index might be preserved from df_sorted or reset.
    # Let's check correctness of group generation
    calc_group = df_sorted.groupby('race_id', sort=False).size().to_numpy()
    
    print(f"Calculated Group (from Sort): {calc_group[:5]}")
    print(f"Dataset Group: {group[:5]}")
    
    if np.array_equal(calc_group, group):
        print("✅ Group arrays match.")
    else:
        print("❌ Group arrays DO NOT match.")
    
    # Validation 3: Check NaN in Features
    print("\nNull Checks in X:")
    null_counts = X.isnull().sum()
    print(null_counts[null_counts > 0])

if __name__ == "__main__":
    debug()

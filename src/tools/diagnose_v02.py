import os
import sys
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.feature_pipeline import FeaturePipeline
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.cleansing import DataCleanser

def diagnose():
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 1000)

    # 1. Load Data (JRA Only)
    print("Loading Data (JRA Only)...")
    loader = JraVanDataLoader()
    # Use same period as experiment
    raw_df = loader.load(history_start_date='2020-01-01', end_date='2024-12-31', jra_only=True)
    
    cleanser = DataCleanser()
    clean_df = cleanser.cleanse(raw_df)
    
    # 2. Load Features
    print("Loading Features...")
    pipeline = FeaturePipeline(cache_dir="data/features")
    # v02 uses base_attributes + history_stats
    df = pipeline.load_features(clean_df, ['base_attributes', 'history_stats'])
    
    # 3. Validation
    print("="*50)
    print("Info")
    print("="*50)
    print(f"Shape: {df.shape}")
    
    # Null Check
    print("\n" + "="*50)
    print("Null Rate Check (history_stats columns)")
    print("="*50)
    hist_cols = ['interval', 'lag1_rank', 'lag1_time_diff', 'mean_rank_5', 'mean_time_diff_5']
    for col in hist_cols:
        if col in df.columns:
            null_rate = df[col].isnull().mean()
            print(f"{col}: {null_rate:.2%}")
        else:
            print(f"{col}: Not Found!")
            
    # Data Inspection
    print("\n" + "="*50)
    print("Data Inspection (Sample Horse)")
    print("="*50)
    
    # Find a horse with many races (top 1)
    # Need to merge date/rank back from clean_df because feature df might only have keys + features
    # Ensure correct join
    merged_check = pd.merge(
        df, 
        clean_df[['race_id', 'horse_number', 'date', 'rank', 'time_diff']], 
        on=['race_id', 'horse_number'], 
        how='left'
    )
    
    top_horse = merged_check['horse_id'].value_counts().index[0]
    print(f"Horse ID: {top_horse}")
    
    sample_df = merged_check[merged_check['horse_id'] == top_horse].sort_values('date')
    
    cols_to_show = ['date', 'race_id', 'rank', 'lag1_rank', 'time_diff', 'lag1_time_diff', 'mean_rank_5']
    print(sample_df[cols_to_show].to_string(index=False))

    # Feature Importance
    print("\n" + "="*50)
    print("Feature Importance")
    print("="*50)
    model_path = "models/experiments/v02_history/model.pkl"
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        importance = model.feature_importance(importance_type='gain')
        feature_names = model.feature_name()
        
        imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
        imp_df = imp_df.sort_values('importance', ascending=False)
        
        print("--- Top 20 Features ---")
        print(imp_df.head(20))
        
        print("\n--- History Feature Ranks ---")
        hist_rows = imp_df[imp_df['feature'].isin(hist_cols)]
        if not hist_rows.empty:
            print(hist_rows)
        else:
            print("History features not found in model!")
    else:
        print(f"Model not found at {model_path}")

if __name__ == "__main__":
    diagnose()

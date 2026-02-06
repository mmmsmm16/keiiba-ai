
import pandas as pd
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = '/workspace'
sys.path.insert(0, PROJECT_ROOT)

from src.inference.preprocessor import InferencePreprocessor

def debug():
    preprocessed_path = os.path.join(PROJECT_ROOT, 'data', 'preprocessed_data.parquet')
    
    if not os.path.exists(preprocessed_path):
        print("Data not found")
        return

    print(f"Loading {preprocessed_path}...")
    history_df = pd.read_parquet(preprocessed_path)
    
    # Simulate the patching logic from v12_realtime_comparison.py
    class_level_cols = ['class_level', 'class_level_n_races', 'class_level_win_rate', 'class_level_top3_rate']
    patched_count = 0
    for col in class_level_cols:
        if col not in history_df.columns:
            print(f"Patching missing column: {col}")
            history_df[col] = 0
            patched_count += 1
        else:
             print(f"Column exists: {col}")
    
    print(f"Patched {patched_count} columns.")
    
    # Select a sample to mimic race_df (new_df)
    # Get a race_id from 2025 Jan
    history_df['date'] = pd.to_datetime(history_df['date'])
    df_2025 = history_df[(history_df['date'].dt.year == 2025) & (history_df['date'].dt.month == 1)].copy()
    
    if df_2025.empty:
        print("No 2025 data found for testing")
        return

    race_id = df_2025['race_id'].iloc[0]
    print(f"Testing with race_id: {race_id}")
    
    race_df = df_2025[df_2025['race_id'] == race_id].copy()
    
    # Preprocess with patched history
    preprocessor = InferencePreprocessor()
    
    # Mocking loader/FeatureEngineer partially by just using the raw row
    # The preprocessor expects raw data usually, but here we pass 'preprocessed' data which is fine-ish
    # Actually InferencePreprocessor expects raw data loaded by loader.
    # But if we pass preprocessed data as 'raw', FeatureEngineer might complain or double-process.
    # However, 'race_df' here is already processed.
    # 'InferencePreprocessor.preprocess' calls 'FeatureEngineer.add_features', then 'HistoryAggregator'.
    # If we pass processed data, 'add_features' might overwrite or just work.
    
    # Let's verify 'history_df' columns strictly before passing
    has_cols = all(col in history_df.columns for col in class_level_cols)
    print(f"History DF has all class_level cols before calling preprocess: {has_cols}")
    
    # We really just want to verify update_incremental_stats logic inside preprocess
    # We can try to modify InferencePreprocessor temporarily OR just trust that if input has cols, it works.
    
    # The error was "Missing columns in history".
    
    # Let's perform the check manually that update_incremental_stats does
    keys = ['class_level']
    metrics = ['n_races', 'win_rate', 'top3_rate']
    prefix = 'class_level'
    hist_cols = keys + [f'{prefix}_{m}' for m in metrics]
    
    for col in hist_cols:
        if col not in history_df.columns:
             print(f"FAIL: {col} is MISSING in history_df")
        else:
             print(f"OK: {col} is PRESENT in history_df")

if __name__ == "__main__":
    debug()

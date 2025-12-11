
import pandas as pd
import os
import sys
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = '/workspace'
sys.path.insert(0, PROJECT_ROOT)

from src.inference.loader import InferenceDataLoader
from src.inference.preprocessor import InferencePreprocessor

def text_profile():
    preprocessed_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'preprocessed_data.parquet')
    
    print(f"Loading history from {preprocessed_path}...")
    t0 = time.time()
    history_df = pd.read_parquet(preprocessed_path)
    # Patch
    class_level_cols = ['class_level', 'class_level_n_races', 'class_level_win_rate', 'class_level_top3_rate']
    for col in class_level_cols:
        if col not in history_df.columns:
            history_df[col] = 0
            
    # Sort
    history_df = history_df.sort_values('date').reset_index(drop=True)
    print(f"History prepared in {time.time() - t0:.2f}s. Shape: {history_df.shape}")
    
    # Get 2025 Jan races
    df_2025 = history_df[(history_df['date'].dt.year == 2025) & (history_df['date'].dt.month == 1)].copy()
    race_ids = df_2025['race_id'].unique()
    print(f"Found {len(race_ids)} races.")
    
    if len(race_ids) == 0:
        return

    loader = InferenceDataLoader()
    preprocessor = InferencePreprocessor()
    
    # Test first 3 races
    for i, race_id in enumerate(race_ids[:3]):
        print(f"\n--- Race {race_id} ---")
        
        t1 = time.time()
        race_df = loader.load(race_ids=[str(race_id)])
        t_load = time.time() - t1
        print(f"Loader: {t_load:.4f}s")
        
        if race_df.empty:
            continue
            
        t2 = time.time()
        race_date = pd.to_datetime(race_df['date'].iloc[0])
        idx = history_df['date'].searchsorted(race_date)
        history_subset = history_df.iloc[:idx]
        t_slice = time.time() - t2
        print(f"History Slice: {t_slice:.4f}s (Subset size: {len(history_subset)})")
        
        t3 = time.time()
        X, ids = preprocessor.preprocess(race_df, history_df=history_subset)
        t_preprocess = time.time() - t3
        print(f"Preprocess: {t_preprocess:.4f}s")

if __name__ == "__main__":
    text_profile()


import pandas as pd
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = '/workspace'
sys.path.insert(0, PROJECT_ROOT)

from src.inference.loader import InferenceDataLoader
from src.inference.preprocessor import InferencePreprocessor
from src.preprocessing.feature_engineering import FeatureEngineer
from src.preprocessing.aggregators import HistoryAggregator

def debug():
    preprocessed_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'preprocessed_data.parquet')
    
    print(f"Loading history from {preprocessed_path}...")
    history_df = pd.read_parquet(preprocessed_path)
    history_df['date'] = pd.to_datetime(history_df['date'])
    
    # Patch cols
    class_level_cols = ['class_level', 'class_level_n_races', 'class_level_win_rate', 'class_level_top3_rate']
    for col in class_level_cols:
        if col not in history_df.columns:
            history_df[col] = 0

    print("History loaded.")
    
    # Pick a race
    df_2025 = history_df[(history_df['date'].dt.year == 2025) & (history_df['date'].dt.month == 1)].copy()
    if df_2025.empty:
        print("No 2025 data!")
        return

    race_id = df_2025['race_id'].unique()[0]
    print(f"Target Race ID: {race_id} (Type: {type(race_id)})")
    
    print("Loading race data via loader...")
    loader = InferenceDataLoader()
    race_df = loader.load(race_ids=[str(race_id)])
    print(f"Loader returned race_df: {race_df.shape}")
    if not race_df.empty:
        print(f"Race DF Race ID Type: {type(race_df['race_id'].iloc[0])}")
        print(f"Race DF Race ID Value: {race_df['race_id'].iloc[0]}")
    else:
        print("Loader returned empty!")
        return

    # Filter history
    race_date = pd.to_datetime(race_df['date'].iloc[0])
    history_subset = history_df[history_df['date'] < race_date]
    print(f"History Subset Size: {history_subset.shape}")
    
    preprocessor = InferencePreprocessor()
    
    # Trace inside preprocess (Simulation)
    new_df = race_df.copy()
    
    # 1. Feature Engineer
    feat_engineer = FeatureEngineer()
    new_df = feat_engineer.add_features(new_df)
    print(f"After FeatureEngineer: {new_df.shape}")
    
    target_ids = new_df['horse_id'].unique()
    relevant_history = history_subset[history_subset['horse_id'].isin(target_ids)].copy()
    print(f"Relevant History Size: {relevant_history.shape}")
    
    combined_horse_df = pd.concat([relevant_history, new_df], axis=0, ignore_index=True)
    print(f"Combined Size: {combined_horse_df.shape}")
    
    # 2. History Aggregator
    hist_agg = HistoryAggregator()
    combined_horse_df = hist_agg.aggregate(combined_horse_df)
    print(f"After HistoryAggregator: {combined_horse_df.shape}")
    
    # 3. Call actual preprocess
    print("Calling preprocessor.preprocess() ...")
    try:
        X, ids = preprocessor.preprocess(race_df, history_df=history_subset)
        print(f"Result X: {X.shape}")
        
        if X.empty:
             print("X is empty! Debugging target_ids matching in final step...")
             target_race_ids = new_df['race_id'].unique()
             combined_race_ids = combined_horse_df['race_id'].unique()
             
             print(f"Target Race IDs (new_df): {target_race_ids}")
             print(f"Type of Target Race ID: {type(target_race_ids[0])}")
             
             # Need to mimic what happened inside preprocess. 
             # Inside preprocess, 'combined_horse_df' is modified by update_incremental_stats which uses 'merge'.
             # If merge failure, maybe rows were dropped? 'how=left' usually preserves rows.
             
             # But let's check if combined_horse_df has the race_id
             # Wait, local variable 'combined_horse_df' here is disconnected from what happened inside 'preprocess'.
             # We should probably debug inside preprocess.
             pass
    except Exception as e:
        print(f"Preprocess failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug()

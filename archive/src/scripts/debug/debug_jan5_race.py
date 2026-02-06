
import pandas as pd
import os
import sys
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = '/workspace'
sys.path.insert(0, PROJECT_ROOT)

from src.inference.loader import InferenceDataLoader
from src.inference.preprocessor import InferencePreprocessor
from src.model.ensemble import EnsembleModel

def debug_race():
    # 1. Get a target race ID from cached predictions
    pred_path = os.path.join(PROJECT_ROOT, 'experiments', 'v12_tabnet_revival', 'reports', 'predictions.parquet')
    pred_df = pd.read_parquet(pred_path)
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    target_race = pred_df[pred_df['date'] == '2025-01-05'].iloc[0]
    race_id = str(target_race['race_id'])
    print(f"Target Race: {race_id} ({target_race['date']})")
    
    # 2. Load History (Correct Path)
    hist_path = os.path.join(PROJECT_ROOT, 'experiments', 'v12_tabnet_revival', 'data', 'preprocessed_data.parquet')
    print(f"Loading history from {hist_path}...")
    history_df = pd.read_parquet(hist_path)
    history_df['date'] = pd.to_datetime(history_df['date'])
    
    # 3. Load Race Data
    loader = InferenceDataLoader()
    race_df = loader.load(race_ids=[race_id])
    print(f"Loaded race df: {race_df.shape}")
    
    # 4. Preprocess
    print("Checking class_level types BEFORE preprocess:")
    if 'class_level' in race_df.columns:
        print(f"Race DF class_level type: {race_df['class_level'].dtype}")
        print(f"Race DF class_level unique: {race_df['class_level'].unique()}")
    
    if 'class_level' in history_df.columns:
        print(f"History DF class_level type: {history_df['class_level'].dtype}")
        print(f"History DF class_level unique (sample): {history_df['class_level'].unique()[:10]}")
        
    preprocessor = InferencePreprocessor()
    print("Preprocessing...")
    X, ids = preprocessor.preprocess(race_df, history_df=history_df)
    
    print(f"Generated X shape: {X.shape}")
    
    # Debug Merge Logic
    print("\n--- Merge Debug ---")
    print(f"IDs horse_number type: {ids['horse_number'].dtype}")
    print(f"IDs sample: {ids['horse_number'].tolist()[:5]}")
    
    if 'odds' in race_df.columns:
        print(f"RaceDF horse_number type: {race_df['horse_number'].dtype}")
        print(f"RaceDF sample: {race_df['horse_number'].tolist()[:5]}")
        
        merged = ids.merge(race_df[['horse_number', 'odds']], on='horse_number', how='left')
        print(f"Merged columns: {merged.columns.tolist()}")
        if 'odds' in merged.columns:
            print(f"Merged odds sample: {merged['odds'].tolist()[:5]}")
            print(f"NaN odds count: {merged['odds'].isna().sum()} / {len(merged)}")
        else:
             print("KeyError 'odds' avoided. Check columns above.")
    else:
        print("Odds column MISSING in race_df")
    print("-------------------\n")

    if X.empty:
        print("X is empty!")
        return

    # 5. Check key features
    # Check Class Level
    cls_cols = [c for c in X.columns if 'class_level' in c]
    print(f"\nClass Level Features: {cls_cols}")
    if cls_cols:
        print(X[cls_cols].describe().T[['mean', 'max', 'min']])

    # Check Bloodline
    bl_cols = [c for c in X.columns if 'sire_' in c or 'bms_' in c]
    print(f"\nBloodline Features (sample): {bl_cols[:5]}")
    if bl_cols:
        print(X[bl_cols[:5]].describe().T[['mean', 'max', 'min']])
        
    # Check Horse History
    hist_cols = ['horse_win_rate', 'horse_avg_rank', 'horse_n_races'] # Adjust names if needed based on actual features
    # Actually look for columns starting with 'horse_' or similar commonly used ones
    # tabnet.features.json likely has 'lag1_rank' etc.
    lag_cols = [c for c in X.columns if 'lag1_' in c]
    print(f"\nLag Features (sample): {lag_cols[:5]}")
    if lag_cols:
         print(X[lag_cols[:5]].describe().T[['mean', 'max', 'min']])

    # 6. Feature Adaptation & Prediction
    # Load expected features
    import json
    feat_path = os.path.join(PROJECT_ROOT, 'experiments', 'v12_tabnet_revival', 'models', 'tabnet.features.json')
    with open(feat_path, 'r') as f:
        expected_features = json.load(f)
    
    # Filter X
    for feat in expected_features:
        if feat not in X.columns:
            X[feat] = 0
    X = X[expected_features]
    print(f"Filtered X shape: {X.shape}")

    model_path = os.path.join(PROJECT_ROOT, 'experiments', 'v12_tabnet_revival', 'models', 'ensemble.pkl')
    try:
        # Correctly load using load_model which handles TabNet separation
        model = EnsembleModel()
        model.load_model(model_path, device_name='cpu')
        
        probs = model.predict(X)
        print(f"\nPredictions (Top 5): {probs[:5]}")
        print(f"Max Prob: {np.max(probs)}")
        print(f"Mean Prob: {np.mean(probs)}")
        
        # Check EV roughly (using dummy odds or just looking at probs)
        # If max prob is 0.26 (matches cached), then we are good.
    except Exception as e:
        print(f"Model load/predict failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_race()

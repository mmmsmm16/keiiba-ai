import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from src.inference.loader import InferenceDataLoader
from src.inference.preprocessor import InferencePreprocessor
from src.model.ensemble import EnsembleModel

def debug_inference(race_id, model_version):
    print(f"--- Debugging for Race ID: {race_id}, Model: {model_version} ---")

    # 1. Check Model Path
    base_model_dir = os.path.join(os.path.dirname(__file__), '../../../models')
    filename = f'ensemble_{model_version}.pkl'
    if model_version == 'v4_2025':
        filename = 'ensemble_v4_2025.pkl'
    elif model_version == 'v5':
        filename = 'ensemble_v5.pkl'
    
    model_path = os.path.join(base_model_dir, filename)
    print(f"Checking model path: {os.path.abspath(model_path)}")
    if os.path.exists(model_path):
        print("  [OK] Model file exists.")
    else:
        print("  [ERROR] Model file NOT found.")
        return

    # 2. Check Data Loading
    print("Loading data...")
    try:
        loader = InferenceDataLoader()
        df = loader.load(race_ids=[race_id])
        if df.empty:
            print("  [ERROR] No data found for this race_id.")
            return
        else:
            print(f"  [OK] Data loaded. Shape: {df.shape}")
            print(f"       Columns: {df.columns.tolist()[:5]}...")
    except Exception as e:
        print(f"  [ERROR] Data loading failed: {e}")
        return

    # 3. Check Preprocessing
    print("Running preprocessing...")
    try:
        preprocessor = InferencePreprocessor()
        X, ids = preprocessor.preprocess(df)
        if X.empty:
            print("  [ERROR] Preprocessing returned empty X.")
            return
        else:
            print(f"  [OK] Preprocessing successful. X shape: {X.shape}")
    except Exception as e:
        print(f"  [ERROR] Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Check Model Loading & Prediction
    print("Loading model and predicting...")
    try:
        model = EnsembleModel()
        model.load_model(model_path)
        print("  [OK] Model loaded.")
        
        scores = model.predict(X)
        print(f"  [OK] Prediction successful. Scores: {scores[:5]}...")
    except Exception as e:
        print(f"  [ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_inference("202509050101", "v5")
    debug_inference("202509050101", "v4_2025")

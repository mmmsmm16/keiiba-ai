
import sys
import os
import pickle
import lightgbm as lgb
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

def list_features():
    model_path = 'models/experiments/exp_t2_deep_lag/model.pkl'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Check if it's sklearn wrapper or booster
    if hasattr(model, 'booster_'):
        features = model.booster_.feature_name()
    else:
        features = model.feature_name()
        
    print(f"Total Features: {len(features)}")
    print("--- Feature List ---")
    for f in features:
        print(f)

if __name__ == "__main__":
    list_features()

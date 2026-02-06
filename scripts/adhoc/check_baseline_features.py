
import os
import sys
import joblib
import pandas as pd

# Paths
BASE_LTR = "models/experiments/exp_lambdarank/model.pkl"
PROD_T2 = "models/experiments/exp_t2_refined_v3/model.pkl"

def check_model(path, name):
    print(f"--- Checking {name} ---")
    if not os.path.exists(path):
        print(f"Model not found: {path}")
        return

    try:
        model = joblib.load(path)
        if hasattr(model, "feature_name"):
            feats = model.feature_name()
        elif hasattr(model, "booster_"):
            feats = model.booster_.feature_name()
        else:
            feats = model.feature_name_
        
        print(f"Total Features: {len(feats)}")
        
        # Check for odds features
        odds_feats = [f for f in feats if 'odds' in f or 'popularity' in f or 'vote' in f]
        print(f"Odds Features ({len(odds_feats)}): {odds_feats}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_model(BASE_LTR, "Baseline LambdaRank")
    check_model(PROD_T2, "Production T2")

import pandas as pd
import lightgbm as lgb
import os
import pickle

def main():
    path = "models/experiments/exp_t2_refined_v3/model.pkl"
    if not os.path.exists(path):
        print(f"Model not found at {path}")
        return

    print(f"Loading model from {path}...")
    with open(path, 'rb') as f:
        model = pickle.load(f)
        
    if hasattr(model, 'bst_'):
        bst = model.bst_
    elif isinstance(model, lgb.Booster):
        bst = model
    else:
        # Sklearn API
        bst = model.booster_
        
    features = bst.feature_name()
    print(f"Total Features: {len(features)}")
    
    target_features = [
        'last_nige_rate', 
        'avg_first_corner_norm', 
        'sire_heavy_win_rate',
        'race_pace_level_3f', # Verified fixed, but double check usage
        'is_sole_leader',      # Verified fixed
        'odds_10min'
    ]
    
    print(f"Checking usage of {len(target_features)} suspect features...")
    used = []
    unused = []
    
    for tf in target_features:
        if tf in features:
            used.append(tf)
        else:
            unused.append(tf)
            
    print(f"USED: {used}")
    print(f"UNUSED: {unused}")

if __name__ == "__main__":
    main()

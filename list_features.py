import pandas as pd
import pickle
import sys
import os

# Add src to path just in case
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def list_features(pickle_path):
    print(f"Loading {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    features = data['train']['X'].columns.tolist()
    print(f"Total Features: {len(features)}")
    
    # Simple heuristics to group
    groups = {}
    for f in features:
        prefix = f.split('_')[0]
        if 'race' in f or 'month' in f or 'day' in f: prefix = 'meta'
        if 'time' in f or 'speed' in f or 'last_3f' in f: prefix = 'speed'
        if 'emb' in f: prefix = 'embedding'
        if 'trend' in f: prefix = 'realtime'
        if 'course' in f or 'dist' in f: prefix = 'track_stats'
        if 'jockey' in f or 'trainer' in f: prefix = 'human_stats'
        if 'sire' in f or 'blood' in f: prefix = 'bloodline'
        
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(f)
        
    for g, feats in groups.items():
        print(f"\n--- Group: {g} ({len(feats)}) ---")
        print(feats[:5], "..." if len(feats)>5 else "")

if __name__ == "__main__":
    list_features('data/processed/lgbm_datasets_v11.pkl')

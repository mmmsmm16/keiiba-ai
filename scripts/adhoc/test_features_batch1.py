
import os
import sys
import pandas as pd
import numpy as np

# Adjust path to import src
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.features.mining_features import MiningFeatureGenerator
from src.preprocessing.features.track_aptitude import TrackAptitudeFeatureGenerator

def test_features_batch1():
    print("Loading data...")
    loader = JraVanDataLoader()
    # Skip training data to speed up
    df = loader.load(limit=2000, history_start_date='2018-01-01', skip_training=True, skip_odds=True)
    df = df.sort_values('date').reset_index(drop=True)
    print(f"Data loaded: {len(df)} rows.")

    print("\n--- Testing Mining Features ---")
    mining_gen = MiningFeatureGenerator()
    df_mining = mining_gen.transform(df)
    print("Mining Feature Columns:", df_mining.columns.tolist())
    print(df_mining.head())
    print("Stats:")
    print(df_mining.describe())

    print("\n--- Testing Track Aptitude Features ---")
    track_gen = TrackAptitudeFeatureGenerator()
    df_track = track_gen.transform(df)
    print("Track Feature Columns:", df_track.columns.tolist())
    print(df_track.tail(10)) # Show tail to see accumulated stats
    
    # Check simple logic:
    # Pick a horse with multiple runs
    top_horses = df['horse_id'].value_counts().head(3).index.tolist()
    for hid in top_horses:
        print(f"\nChecking Horse: {hid}")
        mask = df['horse_id'] == hid
        sub_raw = df.loc[mask, ['date', 'going_code', 'rank']]
        sub_feat = df_track.loc[mask]
        
        comparison = pd.concat([
            sub_raw.reset_index(drop=True), 
            sub_feat.reset_index(drop=True)
        ], axis=1)
        print(comparison)

if __name__ == "__main__":
    test_features_batch1()

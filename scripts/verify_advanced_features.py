
import pandas as pd
import numpy as np
import os
import sys
import logging

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from nar.loader import NarDataLoader
from nar.features import NarFeatureGenerator

logging.basicConfig(level=logging.INFO)

def verify_advanced_features():
    loader = NarDataLoader()
    raw_df = loader.load(limit=20000, region='south_kanto')
    gen = NarFeatureGenerator()
    df = gen.generate_features(raw_df)

    new_features = [
        'gender', 'age', 'days_since_prev_race', 'weight_diff',
        'horse_jockey_place_rate', 'is_consecutive_jockey',
        'distance_diff', 'horse_venue_place_rate',
        'trainer_30d_win_rate'
    ]

    print("\n--- NEW FEATURES VERIFICATION ---")
    for feat in new_features:
        if feat in df.columns:
            null_count = df[feat].isnull().sum()
            sample_vals = df[feat].dropna().unique()[:5]
            print(f"Feature: {feat:<25} | Nulls: {null_count:>5} | Samples: {sample_vals}")
        else:
            print(f"Feature: {feat:<25} | MISSING!")

    # Check gender parsing
    print("\nGender counts:")
    print(df['gender'].value_counts())

    # Check chemistry for a horse that had multiple runs
    print("\nSample Horse-Jockey chemistry:")
    sample_horse = df[df['horse_id'] == df['horse_id'].value_counts().index[0]][['date', 'jockey_id', 'rank', 'horse_jockey_place_rate', 'is_consecutive_jockey']]
    print(sample_horse.head(10))

if __name__ == "__main__":
    verify_advanced_features()


import pandas as pd
import numpy as np
import logging
import sys
import os

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from nar.features import NarFeatureGenerator
from nar.loader import NarDataLoader

logging.basicConfig(level=logging.INFO)

def check_feature_nans():
    loader = NarDataLoader()
    df = loader.load(limit=5000, region='south_kanto')
    gen = NarFeatureGenerator()
    df_feat = gen.generate_features(df)
    
    # Check for NaNs in human stats
    human_cols = ['jockey_win_rate', 'jockey_place_rate', 'trainer_win_rate', 'trainer_place_rate']
    print("\n--- Human Stats NaN Check ---")
    print(df_feat[human_cols].isnull().mean())
    print("\nSample values:")
    print(df_feat[human_cols].dropna().head())

    # Check data types of features used in modeling
    features = [
        'distance', 'venue', 'state', 'frame_number', 'horse_number', 'weight', 'impost',
        'jockey_win_rate', 'jockey_place_rate', 'trainer_win_rate', 'trainer_place_rate',
        'horse_run_count'
    ] + [col for col in df_feat.columns if 'horse_prev' in col]
    
    print("\n--- Feature Data Types ---")
    print(df_feat[features].dtypes)
    
    # Check if 'state' or 'venue' are objects
    print("\nObject columns in features:")
    print(df_feat[features].select_dtypes(include=['object']).columns.tolist())

if __name__ == "__main__":
    check_feature_nans()

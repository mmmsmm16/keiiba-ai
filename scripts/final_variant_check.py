#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline

RACE_ID = "202606010910"
DATE = "2026-01-25"

def main():
    loader = JraVanDataLoader()
    df_raw = loader.load_for_horses(['2018104775'], DATE, "2016-01-01") # Just one horse for quick test
    
    pipeline = FeaturePipeline()
    # Force recalculate to test new track_variant logic
    df_features = pipeline.load_features(df_raw, ['race_conditions', 'rating_elo'], force=True)
    
    today_feat = df_features[df_features['race_id'] == RACE_ID]
    print(f"Target Race track_variant: {today_feat['track_variant'].iloc[0]}")
    print(f"Target Race horse_elo: {today_feat['horse_elo'].iloc[0]}")

if __name__ == "__main__":
    main()

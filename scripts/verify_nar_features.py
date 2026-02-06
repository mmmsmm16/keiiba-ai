
import os
import pandas as pd
import numpy as np
import logging
import sys

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from nar.features import NarFeatureGenerator
from nar.loader import NarDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_speed_index():
    loader = NarDataLoader()
    # Load a small sample
    df = loader.load(limit=10000, region='south_kanto')
    
    if df.empty:
        logger.error("No data loaded.")
        return

    # Generate features
    gen = NarFeatureGenerator(history_windows=[1, 3])
    df_feat = gen.generate_features(df)
    
    print("\n--- Features Summary ---")
    print(df_feat.columns.tolist())
    
    print("\n--- Speed Index Sample ---")
    cols_to_show = ['race_id', 'horse_name', 'venue', 'distance', 'state', 'time', 'speed_index', 'horse_prev1_si_avg', 'horse_prev3_si_avg']
    print(df_feat[cols_to_show].dropna(subset=['speed_index']).head(20))
    
    # Check if horse_prev1_si_avg matches shifted speed_index
    print("\n--- Logic Check: Horse SI History ---")
    horse_id = df_feat['horse_id'].iloc[20] # Pick a horse
    horse_data = df_feat[df_feat['horse_id'] == horse_id][['date', 'speed_index', 'horse_prev1_si_avg']]
    print(f"Horse: {horse_id}")
    print(horse_data)

    # Check if Speed Index distribution looks reasonable (around 80)
    print("\n--- Speed Index Distribution ---")
    print(df_feat['speed_index'].describe())

if __name__ == "__main__":
    verify_speed_index()

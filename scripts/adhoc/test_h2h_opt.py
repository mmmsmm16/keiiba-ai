
import sys
import os
import pandas as pd
import logging
from datetime import datetime

# Root path alignment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline
from src.preprocessing.features.head_to_head import compute_head_to_head

logging.basicConfig(level=logging.INFO)

def test_h2h():
    print("Loading data for verification (2023)...")
    loader = JraVanDataLoader()
    # 1年分で十分に重いかテスト
    df = loader.load(history_start_date="2023-01-01", end_date="2023-12-31")
    
    print(f"Loaded {len(df)} rows.")
    
    start_time = datetime.now()
    print("Running compute_head_to_head...")
    
    df_h2h = compute_head_to_head(df)
    
    elapsed = datetime.now() - start_time
    print(f"Computation finished in {elapsed}")
    
    print("Columns:", df_h2h.columns)
    print("Sample:\n", df_h2h[['vs_rival_win_rate', 'vs_rival_match_count']].describe())
    print("Head:\n", df_h2h.head(10))

if __name__ == "__main__":
    test_h2h()

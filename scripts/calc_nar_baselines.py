
import os
import pandas as pd
import numpy as np
import logging
import sys

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from nar.loader import NarDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    loader = NarDataLoader()
    # Load data for South Kanto
    df = loader.load(limit=200000, region='south_kanto')
    
    if df.empty:
        logger.error("No data loaded.")
        return

    # Filter for valid times
    df = df[df['time'] > 0].copy()
    
    # Analyze grade_code and kyoso_joken_code to identify high class
    print("\ngrade_code counts:")
    print(df['grade_code'].value_counts().head(10))
    
    print("\nkyoso_joken_code counts:")
    print(df['kyoso_joken_code'].value_counts().head(10))

    # Calculate median time per (venue, distance, state)
    baselines = df.groupby(['venue', 'distance', 'state'])['time'].agg(['median', 'count', 'std']).reset_index()
    
    # Save to CSV for reference
    os.makedirs('data/nar', exist_ok=True)
    baselines.to_csv('data/nar/speed_index_baselines.csv', index=False)
    print(f"\nBaselines saved to data/nar/speed_index_baselines.csv. Total entries: {len(baselines)}")
    
    # Show example for Funabashi (43) 1600m
    example = baselines[(baselines['venue'] == 43) & (baselines['distance'] == 1600)]
    print("\nExample: Funabashi (43) 1600m baselines:")
    print(example)

if __name__ == "__main__":
    main()

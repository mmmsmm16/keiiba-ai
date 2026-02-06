
import pandas as pd
import numpy as np
import pytest
import sys
import os
sys.path.append(os.getcwd())
try:
    from src.preprocessing.features.class_stats import compute_class_stats
except ImportError:
    # Try appending src explicitly if running from root
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from preprocessing.features.class_stats import compute_class_stats

def test_class_stats_leakage():
    # Setup dummy data
    # Horse A runs 3 times in Class "005" (1 Win Class)
    # Date 1: Rank 1 (Win) -> Should impact Date 2 stats
    # Date 2: Rank 4 (Lost) -> Should impact Date 3 stats
    # Date 3: Current Race -> Should NOT use Date 3 rank
    
    data = {
        'horse_id': ['h1', 'h1', 'h1'],
        'date': pd.to_datetime(['2022-01-01', '2022-02-01', '2022-03-01']),
        'race_id': ['r1', 'r2', 'r3'],
        'grade_code': [None, None, None],
        'kyoso_joken_code': ['005', '005', '005'], # Same class
        'rank': [1, 4, 1], # Results
        'horse_number': [1, 2, 3]
    }
    df = pd.DataFrame(data)
    
    # Run feature generation
    res = compute_class_stats(df)
    
    # Merge back to check
    df_merged = pd.merge(df, res, on=['race_id', 'horse_id', 'horse_number'], how='left')
    
    # Test Date 1 (2022-01-01)
    # Should have 0 races history (closed='left')
    row1 = df_merged.iloc[0]
    assert row1['hc_n_races_365d'] == 0, "Date 1 should have 0 history"
    
    # Test Date 2 (2022-02-01)
    # History: Date 1 (Rank 1 -> Top3=1)
    row2 = df_merged.iloc[1]
    assert row2['hc_n_races_365d'] == 1, "Date 2 should have 1 past race"
    assert row2['hc_top3_rate_365d'] == 1.0, "Date 2 history should be 100% top3"
    
    # Test Date 3 (2022-03-01)
    # History: Date 1 (Rank 1), Date 2 (Rank 4) -> 1 Top3 in 2 races
    row3 = df_merged.iloc[2]
    assert row3['hc_n_races_365d'] == 2, "Date 3 should have 2 past races"
    assert row3['hc_top3_rate_365d'] == 0.5, "Date 3 history should be 50% top3"

def test_class_stats_class_change():
    # Horse B: 
    # Date 1: Class 005 (Rank 1) -> Promoted
    # Date 2: Class 010 (Rank 5) -> New Class History should be 0!
    
    data = {
        'horse_id': ['h2', 'h2'],
        'date': pd.to_datetime(['2022-01-01', '2022-02-01']),
        'race_id': ['r4', 'r5'],
        'grade_code': [None, None],
        'kyoso_joken_code': ['005', '010'], # Class Change
        'rank': [1, 5], 
        'horse_number': [1, 2]
    }
    df = pd.DataFrame(data)
    
    res = compute_class_stats(df)
    df_merged = pd.merge(df, res, on=['race_id', 'horse_id', 'horse_number'], how='left')
    
    # Date 2: Class 010. 
    # Past history is Class 005. So history for Class 010 should be 0.
    row2 = df_merged.iloc[1]
    assert row2['hc_n_races_365d'] == 0, "History should be class-specific"
    
    # Check Trend
    # is_same_class_prev should be 0 (False)
    assert row2['is_same_class_prev'] == 0, "Class changed, flag should be 0"

if __name__ == "__main__":
    # Manually run if executed as script
    try:
        test_class_stats_leakage()
        test_class_stats_class_change()
        print("✅ tests_class_stats passed!")
    except AssertionError as e:
        print(f"❌ Test Failed: {e}")
        exit(1)

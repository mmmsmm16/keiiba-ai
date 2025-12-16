
import pandas as pd
import numpy as np
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.advanced_features import AdvancedFeatureEngineer

# Mock logger
logging.basicConfig(level=logging.INFO)

def test_speed_index_robust():
    print("Testing Robust Speed Index Calculation...")
    
    # 1. Feature Drop Test & Lag Generation
    # Create N=40 rows (enough for calc)
    rows = []
    # Course 1, Dist 1600, Cond 10
    for i in range(40):
        # Time improving: 100 - i*0.1
        rows.append({
            'race_id': f'R{i}',
            'horse_id': 'H1', # Same horse to check lag shift
            'date': pd.to_datetime('2024-01-01') + pd.Timedelta(days=i),
            'course_id': 1,
            'distance': 1600,
            'track_condition_code': 10,
            'time': 100.0 - (i * 0.1), 
            'last_3f': 35.0,
            'rank': 1
        })
    # Add one row with diff condition (N=1)
    rows.append({
        'race_id': 'R99', 'horse_id': 'H1', 'date': pd.to_datetime('2024-03-01'),
        'course_id': 1, 'distance': 1600, 'track_condition_code': 40, # Bad
        'time': 110.0, 'last_3f': 35.0, 'rank': 1
    })
    
    df = pd.DataFrame(rows)
    print(f"Created DataFrame with {len(df)} rows.")
    
    engineer = AdvancedFeatureEngineer()
    df_result = engineer.add_features(df)
    
    # Check 1: Output Columns Logic
    print("Checking columns...")
    assert 'time_index' not in df_result.columns, "time_index should be dropped!"
    assert 'last_3f_index' not in df_result.columns, "last_3f_index should be dropped!"
    assert 'lag1_time_index' in df_result.columns, "lag1_time_index should exist"
    
    # Check 2: Calculation Logic (N >= 30)
    # The first 30 rows (idx 0 to 29) will have N < 30 (prev count in expanding).
    # because row 0 has prev=0. row 29 has prev=29.
    # So index 0 to 29 should be 50.0 (Neutral).
    # Row 30 (31st race) has prev=30. Should have score.
    
    # Lag logic:
    # df['lag1_time_index'] is Shift(1) of generated index.
    # Generated Index for Row 0 (N=0) -> 50.
    # Generated Index for Row 29 (N=29) -> 50.
    # Generated Index for Row 30 (N=30) -> Calculated Score.
    
    # df['lag1_time_index'] for Row 30 = Generated Index for Row 29 = 50.
    # df['lag1_time_index'] for Row 31 = Generated Index for Row 30 = Calculated Score.
    
    # Let's check Row 31 (if exists 40 rows, 0-39. Row 31 is index 31).
    if len(df) > 31:
        row31 = df_result.iloc[31]
        print(f"Row 31 Lag1 Index: {row31['lag1_time_index']}")
        # Should be calculated score (not 50)
        # Verify it's not default 50.0
        # Wait, if stats variance is small, maybe score is close to 50?
        # Time sequence: 100, 99.9, ...
        # Std is non-zero.
        assert abs(row31['lag1_time_index'] - 50.0) > 0.1, f"Row 31 should have valid score from Row 30. Got {row31['lag1_time_index']}"
        
    # Check 3: Fallback Logic / Small Sample
    # Last row (R99, idx 40). Group (1, 1600, 40) has N=0 prev.
    # So Generated Index = 50.
    # Lag1 for next race would be 50.
    # Current row's lag1 comes from Row 39 (H1 history).
    # Since we sorted by horse_id/date.
    # Row 40 is same horse H1.
    # It should receive Row 39's index as lag1.
    # Row 39 (N=39) has valid index.
    # So Row 40 lag1 should be Valid.
    
    row40 = df_result.iloc[40]
    print(f"Row 40 Lag1 Index: {row40['lag1_time_index']}")
    assert abs(row40['lag1_time_index'] - 50.0) > 0.1, "Row 40 should inherit valid lag1 from Row 39"
    
    print("\nRobust Speed Index Logic Verified: PASS")

if __name__ == "__main__":
    test_speed_index_robust()


import sys
import os
import pandas as pd
import numpy as np

# Adjust path to find src
current_dir = os.getcwd()
sys.path.append(current_dir)
if 'src' not in sys.path:
    sys.path.append(os.path.join(current_dir, 'src'))

print(f"DEBUG: sys.path: {sys.path}")

try:
    from src.preprocessing.features.class_stats import compute_class_stats
    print("DEBUG: Successfully imported compute_class_stats")
except ImportError as e:
    print(f"ERROR: Import failed: {e}")
    sys.exit(1)

def run_debug():
    print("DEBUG: Creating dummy data...")
    data = {
        'horse_id': ['h1', 'h1', 'h1'],
        'date': pd.to_datetime(['2022-01-01', '2022-02-01', '2022-03-01']),
        'race_id': ['r1', 'r2', 'r3'],
        'grade_code': [None, None, None],
        'kyoso_joken_code': ['005', '005', '005'],
        'rank': [1, 4, 1],
        'horse_number': [1, 2, 3]
    }
    df = pd.DataFrame(data)
    
    print("DEBUG: Running compute_class_stats...")
    try:
        res = compute_class_stats(df)
        print("DEBUG: Result shape:", res.shape)
        print("DEBUG: Result columns:", res.columns.tolist())
        print(res.head())
        
        # Check Expected Values
        df_merged = pd.merge(df, res, on=['race_id', 'horse_id', 'horse_number'], how='left')
        print("DEBUG: Validating logic...")
        
        # Test 1
        val1 = df_merged.iloc[0]['hc_n_races_365d']
        print(f"DEBUG: Row 0 hc_n_races_365d = {val1} (Expected 0.0)")
        
        # Test 2
        val2 = df_merged.iloc[1]['hc_n_races_365d']
        print(f"DEBUG: Row 1 hc_n_races_365d = {val2} (Expected 1.0)")
        
        if val1 == 0 and val2 == 1:
            print("✅ SUCCESS: Logic verified.")
        else:
            print("❌ FAILURE: Logic incorrect.")
            
    except Exception as e:
        print(f"ERROR: Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_debug()


import pandas as pd
import pytest
import os

# Forbidden substrings or exact matches
# If any feature contains these, it might be a leak.
FORBIDDEN_KEYWORDS = [
    'time', 'agari', 'last_3f', 'rank', 'chakujun', 
    'odds', 'ninki', 'popularity', 'payout', 'return',
    'pass_', 'passing', 'tansho', 'fukusho'
]

# Exceptions (Safe features that might contain keywords)
# e.g. 'race_time' (timestamp?) -> 'date'?
# 'jockey_win_rate_time_weighted' ?
# We need to be careful.
# 'time_index' is forbidden (it's speed index based on final time).
# 'lag1_time_index' is ALLOWED (past data).
ALLOWLIST_PREFIXES = [
    'lag', 'mean_', 'max_', 'min_', 'std_', 
    'jockey_', 'trainer_', 'sire_', 'bms_', 'owner_',
    'course_', 'distance_', 'venue_', 'frame_', 'horse_',
    'race_number', 'weather', 'state', 'surface', 'sex', 'age', 'weight'
]

def test_strict_feature_leakage():
    # PATH
    target_file = 'data/predictions/v13_oof_2025_clean.parquet'
    if not os.path.exists(target_file):
        pytest.skip(f"Target file not found: {target_file}")
    
    df = pd.read_parquet(target_file)
    cols = df.columns.tolist()
    
    leaks = []
    
    for c in cols:
        # Skip meta columns (Validation Targets)
        if c in ['race_id', 'horse_number', 'frame_number', 'date', 'target', 'pred_prob', 'fold', 'is_train', 'rank', 'odds']:
            continue
            
        # Check forbidden
        is_suspicious = False
        for k in FORBIDDEN_KEYWORDS:
            if k in c:
                is_suspicious = True
                break
        
        if is_suspicious:
            # Check if allowed via prefix (Lag features are okay)
            is_allowed = False
            # If it starts with 'lag', it is past data.
            # But 'lag1_time' is okay. 'time' is not.
            if c.startswith('lag') or c.startswith('mean_') or 'win_rate' in c:
                is_allowed = True
            
            # Specific exceptions
            if c in ['race_number']: is_allowed = True
            
            if not is_allowed:
                leaks.append(c)
                
    # Assert
    assert len(leaks) == 0, f"Found {len(leaks)} potential leakage features: {leaks}"

if __name__ == "__main__":
    # Manual run
    try:
        test_strict_feature_leakage()
        print("PASS: No leaks found.")
    except Exception as e:
        print(f"FAIL: {e}")

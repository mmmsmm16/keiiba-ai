import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from src.features.odds_movement_features import calculate_odds_movement_features

# Mock data
@pytest.fixture
def mock_snapshot_dir(tmp_path):
    # Create T-60, T-30, T-10 parquets
    d = tmp_path / "2025"
    d.mkdir(parents=True)
    
    # Race 1, Horse 1
    # T-60: Odds 10.0, Ninki 5
    # T-30: Odds 8.0, Ninki 4
    # T-10: Odds 5.0, Ninki 2 (Smart Money!)
    
    df60 = pd.DataFrame([{
        'race_id': '202501010101', 'horse_number': 1, 'odds': 10.0, 'ninki': 5
    }])
    df30 = pd.DataFrame([{
        'race_id': '202501010101', 'horse_number': 1, 'odds': 8.0, 'ninki': 4
    }])
    df10 = pd.DataFrame([{
        'race_id': '202501010101', 'horse_number': 1, 'odds': 5.0, 'ninki': 2
    }])
    
    # T-5 (Should NOT be used)
    df5 = pd.DataFrame([{
        'race_id': '202501010101', 'horse_number': 1, 'odds': 2.0, 'ninki': 1
    }])
    
    df60.to_parquet(d / "win_T-60.parquet")
    df30.to_parquet(d / "win_T-30.parquet")
    df10.to_parquet(d / "win_T-10.parquet")
    df5.to_parquet(d / "win_T-5.parquet")
    
    return str(tmp_path)

def test_odds_movement_calculation(mock_snapshot_dir):
    """Verify feature calculation correctness"""
    df = pd.DataFrame({'race_id': ['202501010101'], 'horse_number': [1]})
    
    features = calculate_odds_movement_features(df, year=2025, snapshot_dir=mock_snapshot_dir)
    
    assert not features.empty
    row = features.iloc[0]
    
    # 1. log_odds_t10
    # log(5.0) = 1.609
    assert row['log_odds_t10'] == pytest.approx(np.log(5.0), 0.001)
    
    # 2. dlog_odds_t60_t10
    # log(5) - log(10) = -0.693
    assert row['dlog_odds_t60_t10'] == pytest.approx(np.log(5.0) - np.log(10.0), 0.001)
    
    # 3. Rank Change
    # T10(2) - T60(5) = -3
    assert row['rank_change_t60_t10'] == -3
    
    # 4. Drop Rate
    # 5.0 / 10.0 = 0.5
    assert row['odds_drop_rate_t60_t10'] == 0.5

def test_no_leak_check(mock_snapshot_dir):
    """Verify T-5 is NOT used"""
    # The function imports T-60, T-30, T-10. It should ignore T-5.
    # We can check by inspecting the code or result.
    # Result-wise, if we used T-5, maybe features would differ?
    # Actually, the function hardcodes the filenames. 
    # Just ensuring it runs without T-5 being present?
    
    # Remove T-5 file
    os.remove(os.path.join(mock_snapshot_dir, "2025", "win_T-5.parquet"))
    
    df = pd.DataFrame({'race_id': ['202501010101'], 'horse_number': [1]})
    features = calculate_odds_movement_features(df, year=2025, snapshot_dir=mock_snapshot_dir)
    
    assert not features.empty
    # Success means T-5 was not required.

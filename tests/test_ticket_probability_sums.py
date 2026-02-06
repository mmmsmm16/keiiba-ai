import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from src.probability.ticket_probabilities import compute_ticket_probs

@pytest.fixture
def mock_race_df():
    # 10 horses
    # Frames: 1,1, 2,2, 3,3, 4,4, 5,5 (simplification, real is 1-8)
    # Let's use realistic JRA frames for 10 horses:
    # 1, 2, 3, 4, 5, 6, 7, 7, 8, 8 (Starts from 8 frames)
    # Actually simple mapping: 
    frames = [1, 2, 3, 4, 5, 6, 7, 7, 8, 8]
    # Probabilities summing to 1
    # Skewed: Horse 1 is strong (0.4)
    probs = [0.4, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.025, 0.025]
    
    return pd.DataFrame({
        'horse_number': range(1, 11),
        'frame_number': frames,
        'pred_prob': probs
    })

def test_win_sum(mock_race_df):
    results = compute_ticket_probs(mock_race_df, n_samples=10000, seed=42)
    p_win = results['win']
    assert p_win.sum() == pytest.approx(1.0, 0.01)

def test_place_sum(mock_race_df):
    # 10 horses -> 3 places
    results = compute_ticket_probs(mock_race_df, n_samples=10000, seed=42)
    p_place = results['place']
    # Sum of P(Place) should be roughly 3.0
    assert p_place.sum() == pytest.approx(3.0, 0.05)

def test_umaren_sum(mock_race_df):
    results = compute_ticket_probs(mock_race_df, n_samples=10000, seed=42)
    df_umaren = results['umaren']
    
    # Sum of upper triangle should be 1.0 (since order doesn't matter, i-j is same as j-i)
    # The matrix is symmetric.
    # But wait, compute_ticket_probs returns symmetric matrix P({i,j}).
    # If we sum everything and divide by 2?
    # P(i,j) + P(j,i) = 2 * P({i,j}). 
    # Current implementation: p_umaren[i, j] = probs_exact[i, j] + probs_exact[j, i]
    # So M[i,j] IS the prob of pair {i,j}.
    # M[j,i] is ALSO the prob of pair {i,j}.
    # Sum of unique pairs: sum triangle (excluding diagonal which is 0).
    
    unique_sum = 0
    horses = df_umaren.index
    for i in range(len(horses)):
        for j in range(i + 1, len(horses)):
            unique_sum += df_umaren.iloc[i, j]
            
    assert unique_sum == pytest.approx(1.0, 0.01)

def test_wakuren_sum(mock_race_df):
    results = compute_ticket_probs(mock_race_df, n_samples=10000, seed=42)
    df_wakuren = results['wakuren']
    
    # Frames 1-8.
    # Sum of unique pairs (i<=j).
    # Wakuren includes i=j (same frame).
    
    total_prob = 0
    frames = df_wakuren.index # 1..8
    for i in range(len(frames)):
        for j in range(i, len(frames)): # include diagonal
            total_prob += df_wakuren.iloc[i, j]
            
    assert total_prob == pytest.approx(1.0, 0.01)

def test_reproducibility(mock_race_df):
    res1 = compute_ticket_probs(mock_race_df, n_samples=5000, seed=123)
    res2 = compute_ticket_probs(mock_race_df, n_samples=5000, seed=123)
    res3 = compute_ticket_probs(mock_race_df, n_samples=5000, seed=999)
    
    pd.testing.assert_series_equal(res1['win'], res2['win'])
    assert not np.allclose(res1['win'].values, res3['win'].values) # Different seed -> different

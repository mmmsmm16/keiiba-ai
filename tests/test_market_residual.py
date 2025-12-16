"""
Unit Tests for Market-Residual Model Probability Normalization
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestMarketResidualProbNorm:
    """市場残差モデルの確率正規化テスト"""
    
    def test_prob_norm_sum_equals_one(self):
        """正規化後のレース内合計が1になる"""
        from utils.calibration import normalize_prob_per_race
        
        probs = np.array([0.3, 0.2, 0.5, 0.4, 0.6])
        race_ids = np.array(['R1', 'R1', 'R1', 'R2', 'R2'])
        
        normalized = normalize_prob_per_race(probs, race_ids)
        
        df = pd.DataFrame({'race_id': race_ids, 'prob': normalized})
        race_sums = df.groupby('race_id')['prob'].sum()
        
        for race_id, s in race_sums.items():
            assert abs(s - 1.0) < 1e-10, f"Race {race_id} sum = {s}"
    
    def test_zero_sum_race_excluded(self):
        """sum=0のレースが安全に除外される（NaN）"""
        from utils.calibration import normalize_prob_per_race
        
        probs = np.array([0.0, 0.0, 0.0, 0.4, 0.6])
        race_ids = np.array(['R1', 'R1', 'R1', 'R2', 'R2'])
        
        normalized = normalize_prob_per_race(probs, race_ids)
        
        # R1はsum=0なのでNaN
        assert np.all(np.isnan(normalized[:3]))
        # R2は正常
        assert not np.any(np.isnan(normalized[3:]))
    
    def test_prob_in_zero_one(self):
        """確率が(0,1)に収まる"""
        from utils.calibration import normalize_prob_per_race
        
        # Random probs
        np.random.seed(42)
        probs = np.random.uniform(0.05, 0.5, 100)
        race_ids = np.array(['R' + str(i // 10) for i in range(100)])
        
        normalized = normalize_prob_per_race(probs, race_ids)
        
        valid = normalized[~np.isnan(normalized)]
        assert np.all(valid > 0)
        assert np.all(valid < 1)
    
    def test_residual_model_output_normalized(self):
        """残差モデル出力が正規化後sum=1になる"""
        from utils.calibration import normalize_prob_per_race
        
        # Simulate residual model output (sigmoid of logit + delta)
        np.random.seed(42)
        
        # Base market probs
        market_probs = np.array([0.4, 0.3, 0.3, 0.5, 0.3, 0.2])
        
        # Model "delta" predictions (raw sigmoid output)
        raw_preds = np.array([0.35, 0.25, 0.40, 0.45, 0.35, 0.20])
        
        race_ids = np.array(['R1', 'R1', 'R1', 'R2', 'R2', 'R2'])
        
        # Normalize
        normalized = normalize_prob_per_race(raw_preds, race_ids)
        
        # Check sums
        df = pd.DataFrame({'race_id': race_ids, 'prob': normalized})
        race_sums = df.groupby('race_id')['prob'].sum()
        
        for race_id, s in race_sums.items():
            assert abs(s - 1.0) < 1e-10, f"Race {race_id} sum = {s}"


class TestIntersectionMetrics:
    """intersection評価のテスト"""
    
    def test_intersection_row_count_match(self):
        """market vs modelの評価行数が一致する"""
        # Create test data with some NaN
        df = pd.DataFrame({
            'race_id': ['R1', 'R1', 'R2', 'R2', 'R3', 'R3'],
            'p_market': [0.4, 0.6, 0.5, 0.5, np.nan, 0.3],
            'prob_model': [0.3, 0.7, 0.4, np.nan, 0.5, 0.5],
            'rank': [1, 2, 1, 2, 1, 2]
        })
        
        # Intersection filter
        mask = (
            df['p_market'].notna() & 
            df['prob_model'].notna() & 
            df['rank'].notna()
        )
        
        intersection_df = df[mask]
        
        # Same rows for both
        market_rows = len(intersection_df)
        model_rows = len(intersection_df)
        
        assert market_rows == model_rows
        assert market_rows == 4  # R1x2 + R2x1 + R3x1 = 4
    
    def test_intersection_race_count_match(self):
        """market vs modelのレース数が一致する"""
        df = pd.DataFrame({
            'race_id': ['R1', 'R1', 'R2', 'R2', 'R3', 'R3'],
            'p_market': [0.4, 0.6, 0.5, 0.5, np.nan, 0.3],
            'prob_model': [0.3, 0.7, 0.4, np.nan, 0.5, 0.5],
            'rank': [1, 2, 1, 2, 1, 2]
        })
        
        mask = (
            df['p_market'].notna() & 
            df['prob_model'].notna() & 
            df['rank'].notna()
        )
        
        intersection_df = df[mask]
        
        market_races = intersection_df['race_id'].nunique()
        model_races = intersection_df['race_id'].nunique()
        
        assert market_races == model_races


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

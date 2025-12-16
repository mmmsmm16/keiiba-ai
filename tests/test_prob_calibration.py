"""
Unit Tests for Probability Normalization and Calibration
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestProbNormalization:
    """確率正規化のテスト"""
    
    def test_normalization_sum_equals_one(self):
        """正規化後のレース内合計が1になる"""
        from utils.calibration import normalize_prob_per_race
        
        probs = np.array([0.3, 0.2, 0.5, 0.4, 0.6])
        race_ids = np.array(['R1', 'R1', 'R1', 'R2', 'R2'])
        
        normalized = normalize_prob_per_race(probs, race_ids)
        
        # Race R1: sum = 1.0, Race R2: sum = 1.0
        df = pd.DataFrame({'race_id': race_ids, 'prob': normalized})
        race_sums = df.groupby('race_id')['prob'].sum()
        
        for race_id, s in race_sums.items():
            assert abs(s - 1.0) < 1e-10, f"Race {race_id} sum = {s}"
    
    def test_zero_sum_race_handled(self):
        """sum=0のレースが安全に処理される"""
        from utils.calibration import normalize_prob_per_race
        
        probs = np.array([0.0, 0.0, 0.0, 0.4, 0.6])
        race_ids = np.array(['R1', 'R1', 'R1', 'R2', 'R2'])
        
        normalized = normalize_prob_per_race(probs, race_ids)
        
        # R1はsum=0なのでNaN
        assert np.all(np.isnan(normalized[:3]))
        # R2は正常
        assert not np.any(np.isnan(normalized[3:]))
    
    def test_negative_probs_preserved(self):
        """負の確率も処理される（NaNにならない）"""
        from utils.calibration import normalize_prob_per_race
        
        # Edge case: negative probs (should still normalize)
        probs = np.array([0.5, 0.3, 0.2])
        race_ids = np.array(['R1', 'R1', 'R1'])
        
        normalized = normalize_prob_per_race(probs, race_ids)
        assert not np.any(np.isnan(normalized))


class TestFullBetaCalibration:
    """Full Beta校正のテスト"""
    
    def test_no_nan_output(self):
        """出力にNaNが含まれない"""
        from utils.calibration import FullBetaCalibrator
        
        np.random.seed(42)
        probs = np.random.uniform(0.01, 0.99, 1000)
        y_true = (np.random.random(1000) < probs).astype(int)
        
        cal = FullBetaCalibrator()
        cal.fit(probs, y_true)
        
        calibrated = cal.predict(probs)
        
        assert not np.any(np.isnan(calibrated))
        assert not np.any(np.isinf(calibrated))
    
    def test_output_in_zero_one(self):
        """出力が(0,1)に収まる"""
        from utils.calibration import FullBetaCalibrator
        
        np.random.seed(42)
        probs = np.random.uniform(0.01, 0.99, 1000)
        y_true = (np.random.random(1000) < probs).astype(int)
        
        cal = FullBetaCalibrator()
        cal.fit(probs, y_true)
        
        calibrated = cal.predict(probs)
        
        assert np.all(calibrated > 0)
        assert np.all(calibrated < 1)
    
    def test_extreme_probs_handled(self):
        """極端な確率値も処理される"""
        from utils.calibration import FullBetaCalibrator
        
        probs = np.array([0.001, 0.5, 0.999])
        y_true = np.array([0, 1, 1])
        
        cal = FullBetaCalibrator()
        cal.fit(probs, y_true)
        
        calibrated = cal.predict(probs)
        
        assert not np.any(np.isnan(calibrated))
        assert np.all(calibrated > 0)
        assert np.all(calibrated < 1)


class TestBlendNormalization:
    """ブレンド正規化のテスト"""
    
    def test_blend_sum_equals_one(self):
        """ブレンド後のレース内合計が1になる"""
        from utils.calibration import MarketBlend
        
        p_market = np.array([0.5, 0.3, 0.2, 0.4, 0.6])
        p_model = np.array([0.4, 0.4, 0.2, 0.3, 0.7])
        race_ids = np.array(['R1', 'R1', 'R1', 'R2', 'R2'])
        
        blender = MarketBlend(lambda_=0.5)
        p_blend = blender.blend(p_market, p_model, normalize_per_race=True, race_ids=race_ids)
        
        df = pd.DataFrame({'race_id': race_ids, 'prob': p_blend})
        race_sums = df.groupby('race_id')['prob'].sum()
        
        for race_id, s in race_sums.items():
            assert abs(s - 1.0) < 1e-10, f"Race {race_id} sum = {s}"
    
    def test_lambda_zero_equals_market(self):
        """λ=0でp_marketと一致"""
        from utils.calibration import MarketBlend
        
        p_market = np.array([0.5, 0.3, 0.2])
        p_model = np.array([0.4, 0.4, 0.2])
        
        blender = MarketBlend(lambda_=0.0)
        p_blend = blender.blend(p_market, p_model, normalize_per_race=False)
        
        np.testing.assert_array_almost_equal(p_blend, p_market, decimal=10)
    
    def test_lambda_one_equals_model(self):
        """λ=1でp_modelと一致"""
        from utils.calibration import MarketBlend
        
        p_market = np.array([0.5, 0.3, 0.2])
        p_model = np.array([0.4, 0.4, 0.2])
        
        blender = MarketBlend(lambda_=1.0)
        p_blend = blender.blend(p_market, p_model, normalize_per_race=False)
        
        np.testing.assert_array_almost_equal(p_blend, p_model, decimal=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit Tests for Softmax Probabilities and Race NLL
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestSoftmaxProbabilities:
    """Softmax確率のテスト"""
    
    def test_softmax_sum_equals_one(self):
        """softmaxのレース内合計が1になる"""
        from phase6.train_market_residual_wf import softmax_per_race
        
        logits = np.array([1.0, 2.0, 0.5, 3.0, 1.5, 2.5])
        race_ids = np.array(['R1', 'R1', 'R1', 'R2', 'R2', 'R2'])
        
        probs = softmax_per_race(logits, race_ids)
        
        df = pd.DataFrame({'race_id': race_ids, 'prob': probs})
        race_sums = df.groupby('race_id')['prob'].sum()
        
        for race_id, s in race_sums.items():
            assert abs(s - 1.0) < 1e-10, f"Race {race_id} sum = {s}"
    
    def test_softmax_probs_in_zero_one(self):
        """softmax確率が(0,1)に収まる"""
        from phase6.train_market_residual_wf import softmax_per_race
        
        logits = np.array([10.0, -10.0, 0.0, 5.0, -5.0])
        race_ids = np.array(['R1', 'R1', 'R1', 'R2', 'R2'])
        
        probs = softmax_per_race(logits, race_ids)
        
        assert np.all(probs > 0)
        assert np.all(probs < 1)
    
    def test_softmax_numerical_stability(self):
        """大きなlogitでも数値的に安定"""
        from phase6.train_market_residual_wf import softmax_per_race
        
        # Large logits that would overflow exp() without shifting
        logits = np.array([1000.0, 1001.0, 999.0, 500.0, 501.0])
        race_ids = np.array(['R1', 'R1', 'R1', 'R2', 'R2'])
        
        probs = softmax_per_race(logits, race_ids)
        
        # Should not have any NaN or Inf
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))
        
        # Should still sum to 1
        df = pd.DataFrame({'race_id': race_ids, 'prob': probs})
        race_sums = df.groupby('race_id')['prob'].sum()
        for s in race_sums:
            assert abs(s - 1.0) < 1e-10
    
    def test_softmax_with_temperature(self):
        """温度パラメータでsoftmaxが調整される"""
        from phase6.train_market_residual_wf import softmax_per_race
        
        logits = np.array([2.0, 1.0, 0.0])
        race_ids = np.array(['R1', 'R1', 'R1'])
        
        probs_t1 = softmax_per_race(logits, race_ids, temperature=1.0)
        probs_t2 = softmax_per_race(logits, race_ids, temperature=2.0)
        
        # Higher temperature -> more uniform
        assert probs_t2[0] < probs_t1[0]  # Max prob decreases
        assert probs_t2[2] > probs_t1[2]  # Min prob increases


class TestRaceNLL:
    """Race NLLのテスト"""
    
    def test_race_nll_basic(self):
        """基本的なRace NLL計算"""
        from phase6.train_market_residual_wf import compute_race_nll
        
        df = pd.DataFrame({
            'race_id': ['R1', 'R1', 'R2', 'R2'],
            'rank': [1, 2, 1, 2],
            'prob': [0.6, 0.4, 0.8, 0.2]
        })
        
        nll = compute_race_nll(df, 'prob')
        
        # Expected: mean(-log(0.6), -log(0.8))
        expected = (-np.log(0.6) + -np.log(0.8)) / 2
        
        assert abs(nll - expected) < 1e-10
    
    def test_race_nll_perfect_prediction(self):
        """完璧な予測でNLLが小さくなる"""
        from phase6.train_market_residual_wf import compute_race_nll
        
        df = pd.DataFrame({
            'race_id': ['R1', 'R1', 'R2', 'R2'],
            'rank': [1, 2, 1, 2],
            'prob_good': [0.99, 0.01, 0.95, 0.05],
            'prob_bad': [0.1, 0.9, 0.2, 0.8]
        })
        
        nll_good = compute_race_nll(df, 'prob_good')
        nll_bad = compute_race_nll(df, 'prob_bad')
        
        assert nll_good < nll_bad
    
    def test_race_nll_empty_winners(self):
        """勝者がいない場合NaN"""
        from phase6.train_market_residual_wf import compute_race_nll
        
        df = pd.DataFrame({
            'race_id': ['R1', 'R1'],
            'rank': [2, 3],
            'prob': [0.5, 0.5]
        })
        
        nll = compute_race_nll(df, 'prob')
        
        assert np.isnan(nll)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

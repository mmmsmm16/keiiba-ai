"""
Unit Tests for Risk Metrics
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.risk_metrics import (
    compute_equity_curve,
    compute_max_drawdown,
    compute_max_drawdown_from_transactions,
    validate_max_dd
)


class TestComputeEquityCurve:
    """compute_equity_curve のテスト"""
    
    def test_empty_transactions(self):
        """空の取引リスト"""
        equity = compute_equity_curve([], initial_bankroll=100000)
        assert len(equity) == 1
        assert equity.iloc[0] == 100000
    
    def test_positive_profits_only(self):
        """利益のみの場合"""
        profits = [1000, 2000, 3000]
        equity = compute_equity_curve(profits, initial_bankroll=100000)
        
        assert len(equity) == 4
        assert equity.iloc[0] == 100000
        assert equity.iloc[1] == 101000
        assert equity.iloc[2] == 103000
        assert equity.iloc[3] == 106000
    
    def test_negative_profits_only(self):
        """損失のみの場合"""
        profits = [-1000, -2000, -3000]
        equity = compute_equity_curve(profits, initial_bankroll=100000)
        
        assert len(equity) == 4
        assert equity.iloc[0] == 100000
        assert equity.iloc[1] == 99000
        assert equity.iloc[2] == 97000
        assert equity.iloc[3] == 94000
    
    def test_bankrupt_stops_trading(self):
        """破産停止のテスト"""
        # 100000 - 50000 - 60000 = -10000 (破産)
        profits = [-50000, -60000, 10000]
        equity = compute_equity_curve(profits, initial_bankroll=100000, stop_if_bankrupt=True)
        
        assert len(equity) == 4
        assert equity.iloc[0] == 100000
        assert equity.iloc[1] == 50000
        assert equity.iloc[2] == 0.0  # 破産でクリップ
        assert equity.iloc[3] == 0.0  # 以降もbet=0で0のまま
    
    def test_no_bankrupt_stop(self):
        """破産停止なしの場合（負の資金を許容）"""
        profits = [-50000, -60000, 10000]
        equity = compute_equity_curve(profits, initial_bankroll=100000, stop_if_bankrupt=False)
        
        assert len(equity) == 4
        assert equity.iloc[0] == 100000
        assert equity.iloc[1] == 50000
        assert equity.iloc[2] == -10000  # 負も許容
        assert equity.iloc[3] == 0  # 回復
    
    def test_invalid_bankroll_raises(self):
        """初期資金0以下はエラー"""
        with pytest.raises(ValueError):
            compute_equity_curve([1000], initial_bankroll=0)
        
        with pytest.raises(ValueError):
            compute_equity_curve([1000], initial_bankroll=-100)
    
    def test_pandas_series_input(self):
        """pandas Seriesを入力できる"""
        profits = pd.Series([1000, -500, 2000])
        equity = compute_equity_curve(profits, initial_bankroll=10000)
        
        assert len(equity) == 4
        assert equity.iloc[3] == 12500


class TestComputeMaxDrawdown:
    """compute_max_drawdown のテスト"""
    
    def test_monotonic_increase(self):
        """単調増加 → DD = 0"""
        equity = pd.Series([100, 110, 120, 130])
        max_dd = compute_max_drawdown(equity)
        
        assert max_dd == 0.0
    
    def test_single_drop_and_recover(self):
        """一度下落して回復"""
        # 100 → 80 (DD=20%) → 120 (回復)
        equity = pd.Series([100, 80, 120])
        max_dd = compute_max_drawdown(equity)
        
        assert abs(max_dd - 0.20) < 0.001  # 20%ドローダウン
    
    def test_multiple_drops(self):
        """複数回の下落"""
        # 100 → 90 (DD=10%) → 100 → 70 (DD=30%) → 100
        equity = pd.Series([100, 90, 100, 70, 100])
        max_dd = compute_max_drawdown(equity)
        
        assert abs(max_dd - 0.30) < 0.001  # 最大は30%
    
    def test_total_loss(self):
        """完全な損失（資金0）"""
        equity = pd.Series([100, 50, 0])
        max_dd = compute_max_drawdown(equity)
        
        assert max_dd == 1.0  # 100%ドローダウン
    
    def test_always_bounded_0_to_1(self):
        """どんな入力でも0〜1の範囲"""
        # 極端なケース
        test_cases = [
            [100, 0, 0, 0],
            [1, 1, 1, 1],
            [1000000, 1],
            [0.001, 0.0001],
        ]
        
        for case in test_cases:
            max_dd = compute_max_drawdown(pd.Series(case))
            assert 0.0 <= max_dd <= 1.0, f"Failed for case {case}: max_dd={max_dd}"
    
    def test_empty_equity(self):
        """空の資金曲線"""
        max_dd = compute_max_drawdown(pd.Series([], dtype=float))
        assert max_dd == 0.0
    
    def test_no_zero_division(self):
        """ゼロ割りが起きない"""
        # すべて0のケース
        equity = pd.Series([0, 0, 0])
        max_dd = compute_max_drawdown(equity)
        
        assert max_dd == 0.0  # 定義上0扱い
        assert not np.isnan(max_dd)
        assert not np.isinf(max_dd)


class TestComputeMaxDrawdownFromTransactions:
    """compute_max_drawdown_from_transactions のテスト"""
    
    def test_basic_usage(self):
        """基本的な使用法"""
        # 100000 → 110000 → 100000 → 90000 → 100000
        profits = [10000, -10000, -10000, 10000]
        max_dd, equity = compute_max_drawdown_from_transactions(
            profits, initial_bankroll=100000
        )
        
        # Peak at 110000, min at 90000 → DD = 20000/110000 ≈ 0.182
        assert 0.15 <= max_dd <= 0.20
        assert len(equity) == 5
        assert equity.iloc[0] == 100000
    
    def test_max_dd_never_exceeds_1(self):
        """Max DDは絶対に1.0を超えない"""
        # 大量の損失でも破産停止で1.0以下
        profits = [-50000] * 10  # 合計-500000の損失
        max_dd, equity = compute_max_drawdown_from_transactions(
            profits, initial_bankroll=100000, stop_if_bankrupt=True
        )
        
        assert max_dd <= 1.0


class TestValidateMaxDd:
    """validate_max_dd のテスト"""
    
    def test_valid_dd(self):
        assert validate_max_dd(0.5) is True
        assert validate_max_dd(0.0) is True
        assert validate_max_dd(1.0) is True
    
    def test_invalid_dd(self):
        assert validate_max_dd(1.5) is False
        assert validate_max_dd(-0.1) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit Tests for Bankroll Constraint
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtest.multi_ticket_backtest_v2 import apply_bankroll_constraint


class TestApplyBankrollConstraint:
    """apply_bankroll_constraint のテスト"""
    
    def test_within_constraint(self):
        """制約内なのでそのまま"""
        planned = 1000
        equity = 100000
        max_frac = 0.05  # 5% = 5000円まで
        
        executed, ratio, skipped = apply_bankroll_constraint(
            planned, equity, max_frac, 100, 'scale'
        )
        
        assert executed == 1000
        assert ratio == 1.0
        assert skipped is False
    
    def test_rescale_when_over_constraint(self):
        """制約超過で縮小"""
        planned = 10000
        equity = 100000
        max_frac = 0.05  # 5% = 5000円まで
        
        executed, ratio, skipped = apply_bankroll_constraint(
            planned, equity, max_frac, 100, 'scale'
        )
        
        assert executed <= equity * max_frac  # 制約守る
        assert executed <= planned  # 元より多くならない
        assert ratio < 1.0  # 縮小された
        assert skipped is False
        assert executed % 100 == 0  # 100円単位
    
    def test_skip_mode(self):
        """skipモードで制約超過"""
        planned = 10000
        equity = 100000
        max_frac = 0.05
        
        executed, ratio, skipped = apply_bankroll_constraint(
            planned, equity, max_frac, 100, 'skip'
        )
        
        assert executed == 0
        assert skipped is True
    
    def test_below_min_equity_threshold(self):
        """最低資金未満で停止"""
        planned = 1000
        equity = 50  # 閾値100未満
        
        executed, ratio, skipped = apply_bankroll_constraint(
            planned, equity, 0.05, 100, 'scale'
        )
        
        assert executed == 0
        assert skipped is True
    
    def test_100_yen_rounding(self):
        """100円単位丸め"""
        planned = 5000
        equity = 3333  # 5% = 166.65円
        
        executed, ratio, skipped = apply_bankroll_constraint(
            planned, equity, 0.05, 100, 'scale'
        )
        
        assert executed % 100 == 0
        assert executed <= equity * 0.05
    
    def test_never_exceeds_max_allowed(self):
        """どんな場合も制約を超えない"""
        test_cases = [
            (12000, 100000, 0.05),
            (50000, 10000, 0.10),
            (1000, 5000, 0.01),
        ]
        
        for planned, equity, max_frac in test_cases:
            executed, ratio, skipped = apply_bankroll_constraint(
                planned, equity, max_frac, 100, 'scale'
            )
            
            if not skipped:
                assert executed <= equity * max_frac * 1.01, \
                    f"Failed: executed={executed}, max_allowed={equity * max_frac}"
    
    def test_equity_never_goes_negative_concept(self):
        """資金が負にならない前提でのbet"""
        # bet額 <= equity なので、損失は最大bet額
        # 1レースで資金が負になることはない
        equity = 1000
        planned = 500
        
        executed, ratio, skipped = apply_bankroll_constraint(
            planned, equity, 0.5, 100, 'scale'
        )
        
        assert executed <= equity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

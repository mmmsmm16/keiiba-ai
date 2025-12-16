# Phase 5: Purchase Optimization Report

**Date**: 2025-12-15 16:32

## 1. Acceptance Criteria

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Min Fold ROI | > 50.0% | 最悪年の足切り |
| Max Drawdown | < 30.0% | 破産リスク足切り |

## 2. Best Strategy

| Metric | Value |
|--------|-------|
| **Strategy Name** | `ev0.00_maxO50_k0.20` |
| **Mean ROI** | 70.79% |
| **Std ROI** | 4.97% |
| **Min Fold ROI** | 63.86% |
| **Max Drawdown** | 5550.07% |
| **ROI (Slippage 0.95)** | 66.86% |
| **ROI (Slippage 0.90)** | 62.59% |
| **Bet/Race** | 3.49 |

### Configuration

```yaml
bankroll: 100000
ev_threshold: 0.0
kelly_fraction: 0.2
max_bet_pct: 0.05
max_odds: 50

```

### Fold-wise Results

| Year | ROI (1.0) | ROI (0.95) | ROI (0.90) | Bets | Profit |
|------|-----------|------------|------------|------|--------|
| 2021 | 73.24% | 69.22% | 64.95% | 12,480 | ¥-2,406,140 |
| 2022 | 63.86% | 60.40% | 56.37% | 11,777 | ¥-3,224,080 |
| 2023 | 75.27% | 70.96% | 66.46% | 11,978 | ¥-2,325,990 |

## 3. Top 10 Candidates (After Acceptance Criteria)

| Rank | Strategy | Mean ROI | Std | Min Fold | Max DD |
|------|----------|----------|-----|----------|--------|
| 1 | ev0.00_maxO50_k0.20 | 70.79% | 4.97% | 63.86% | 5550.07% |
| 2 | ev0.00_maxO50_k0.20 | 70.79% | 4.97% | 63.86% | 5550.07% |
| 3 | ev0.00_minO2.0_maxO50_k0.20 | 70.79% | 4.97% | 63.86% | 5550.07% |
| 4 | ev0.00_minO2.0_maxO50_k0.20 | 70.79% | 4.97% | 63.86% | 5550.07% |
| 5 | ev0.00_minO3.0_maxO50_k0.20 | 70.79% | 4.97% | 63.86% | 5550.07% |
| 6 | ev0.00_minO3.0_maxO50_k0.20 | 70.79% | 4.97% | 63.86% | 5550.07% |
| 7 | ev0.00_maxO50_k0.20 | 70.78% | 4.83% | 63.98% | 5393.06% |
| 8 | ev0.00_minO2.0_maxO50_k0.20 | 70.78% | 4.83% | 63.98% | 5393.06% |
| 9 | ev0.00_minO3.0_maxO50_k0.20 | 70.78% | 4.83% | 63.98% | 5393.06% |
| 10 | ev0.05_maxO50_k0.20 | 70.78% | 4.97% | 63.86% | 5540.75% |

## 4. Adoption Decision Rule

```
1. Filter: Min Fold ROI > 50%
2. Filter: Max Drawdown < 30%
3. Select: Maximum Mean ROI
4. Tiebreaker: Minimum Std
```


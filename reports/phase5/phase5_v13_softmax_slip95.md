# Phase 5: Win Optimization - Probability Column Comparison

**Date**: 2025-12-16 03:27
**Strategy Params**:
- Min EV: 1.2
- Max Odds: 50.0
- Min Prob: 0.05
- Top N: 3
- Bet Amount: ¥100

**Odds Configuration** (IMPORTANT):
- Odds Source: `final`
- Slippage Factor: 0.95
- Allow Final Odds: True
- Payout Match Rate: 100%

> ⚠️ WARNING: Using final odds which match confirmed payouts. This may not reflect real executable odds.


## Summary Comparison

| Prob Column | Bets | Hit% | ROI | MaxDD | MinFold ROI | AvgOdds | AvgEV |
|-------------|------|------|-----|-------|-------------|---------|-------|
| prob_residual_softmax | 11,443 | 36.4% | **227.4%** | ¥1,883 | 220.5% | 8.6 | 2.07 |

**Best ROI**: `prob_residual_softmax` (227.4%)
**Best MinFold ROI**: `prob_residual_softmax` (220.5%)

## Yearly Breakdown

### prob_residual_softmax

| Year | Bets | Cost | Return | ROI |
|------|------|------|--------|-----|
| 2022 | 1321 wins | ¥369,200 | ¥814,045 | 220.5% |
| 2023 | 1462 wins | ¥397,300 | ¥921,860 | 232.0% |
| 2024 | 1386 wins | ¥377,800 | ¥866,399 | 229.3% |

## Sanity Checks (Win)

**Status**: ✅ ALL PASSED

| Check | Status | Details |
|-------|--------|----------|
| win_payout_integrity | ✅ PASS | 全ルールPASS (Hit Rate: 36.4%) |
| odds_vs_payout_separation | ✅ PASS | WARNING: payout/bet = odds が100%一致 (単勝では正常だがodds時刻... |
| ev_definition | ✅ PASS | EV = prob * odds (誤差0.01以内) |
| p_market_recomputation | ✅ PASS | p_market = (1/odds)/Σ(1/odds) 一致 (max_diff=1.19e-0... |

### Odds vs Payout Details

**Distribution:**
| Statistic | Odds | Payout |
|-----------|------|--------|
| Min | 1.2 | 0 |
| Median | 6.2 | 0 |
| P95 | 24.1 | 1093 |
| Max | 49.7 | 4864 |

**Payout/Bet vs Odds Match Rate**: 100.0%

**Diff (payout/bet - odds) Distribution:**
- Min: 0.0000
- Median: 0.0000
- Mean: 0.0000
- P95: 0.0000
- Max: 0.0000
- Zero Rate (|diff|<0.01): 100.0%

### p_market Recomputation

**p_market = (1/odds) / Σ(1/odds) Check:**
- Max Diff: 1.19e-07
- P99 Diff: 2.98e-08
- Mean Diff: 4.11e-09
- Above Tolerance Rate: 0.00%
- Total Rows: 128321
- Total Races: 9454

## Recommendation

Based on the comparison:
- **Recommended prob column**: `prob_residual_softmax`
- **Expected ROI**: 227.4%
- **Risk (MaxDD)**: ¥1,883

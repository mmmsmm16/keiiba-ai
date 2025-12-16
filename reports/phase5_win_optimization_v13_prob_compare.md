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
- Slippage Factor: 1.0
- Allow Final Odds: True
- Payout Match Rate: 100%

> ⚠️ WARNING: Using final odds which match confirmed payouts. This may not reflect real executable odds.


## Summary Comparison

| Prob Column | Bets | Hit% | ROI | MaxDD | MinFold ROI | AvgOdds | AvgEV |
|-------------|------|------|-----|-------|-------------|---------|-------|
| prob_residual_softmax | 12,492 | 36.5% | **230.2%** | ¥2,120 | 225.1% | 8.7 | 2.09 |

**Best ROI**: `prob_residual_softmax` (230.2%)
**Best MinFold ROI**: `prob_residual_softmax` (225.1%)

## Yearly Breakdown

### prob_residual_softmax

| Year | Bets | Cost | Return | ROI |
|------|------|------|--------|-----|
| 2022 | 1465 wins | ¥405,300 | ¥912,139 | 225.1% |
| 2023 | 1587 wins | ¥432,000 | ¥1,012,380 | 234.3% |
| 2024 | 1510 wins | ¥411,900 | ¥951,070 | 230.9% |

## Sanity Checks (Win)

**Status**: ✅ ALL PASSED

| Check | Status | Details |
|-------|--------|----------|
| win_payout_integrity | ✅ PASS | 全ルールPASS (Hit Rate: 36.5%) |
| odds_vs_payout_separation | ✅ PASS | WARNING: payout/bet = odds が100%一致 (単勝では正常だがodds時刻... |
| ev_definition | ✅ PASS | EV = prob * odds (誤差0.01以内) |
| p_market_recomputation | ✅ PASS | p_market = (1/odds)/Σ(1/odds) 一致 (max_diff=1.19e-0... |

### Odds vs Payout Details

**Distribution:**
| Statistic | Odds | Payout |
|-----------|------|--------|
| Min | 1.3 | 0 |
| Median | 6.2 | 0 |
| P95 | 24.4 | 1110 |
| Max | 50.0 | 4810 |

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
- Mean Diff: 2.39e-09
- Above Tolerance Rate: 0.00%
- Total Rows: 128321
- Total Races: 9454

## Recommendation

Based on the comparison:
- **Recommended prob column**: `prob_residual_softmax`
- **Expected ROI**: 230.2%
- **Risk (MaxDD)**: ¥2,120

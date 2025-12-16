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
- Slippage Factor: 0.9
- Allow Final Odds: True
- Payout Match Rate: 100%

> ⚠️ WARNING: Using final odds which match confirmed payouts. This may not reflect real executable odds.


## Summary Comparison

| Prob Column | Bets | Hit% | ROI | MaxDD | MinFold ROI | AvgOdds | AvgEV |
|-------------|------|------|-----|-------|-------------|---------|-------|
| prob_residual_softmax | 10,352 | 36.5% | **225.5%** | ¥1,649 | 218.3% | 8.5 | 2.05 |

**Best ROI**: `prob_residual_softmax` (225.5%)
**Best MinFold ROI**: `prob_residual_softmax` (218.3%)

## Yearly Breakdown

### prob_residual_softmax

| Year | Bets | Cost | Return | ROI |
|------|------|------|--------|-----|
| 2022 | 1189 wins | ¥333,700 | ¥728,558 | 218.3% |
| 2023 | 1328 wins | ¥359,200 | ¥827,513 | 230.4% |
| 2024 | 1259 wins | ¥342,300 | ¥778,346 | 227.4% |

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
| P95 | 23.6 | 1071 |
| Max | 49.9 | 4869 |

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
- Mean Diff: 3.91e-09
- Above Tolerance Rate: 0.00%
- Total Rows: 128321
- Total Races: 9454

## Recommendation

Based on the comparison:
- **Recommended prob column**: `prob_residual_softmax`
- **Expected ROI**: 225.5%
- **Risk (MaxDD)**: ¥1,649

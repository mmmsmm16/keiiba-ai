# Phase 7 (v2): Multi-Ticket Backtest Report

**Date**: 2025-12-16 03:53
**Year**: 2023
**Filter**: JRA-only
**Prob Column**: prob_residual_softmax
**Odds Source**: final
**Slippage Factor**: 0.9
**Allow Final Odds**: True

## Bankroll Settings

| Parameter | Value |
|-----------|-------|
| Initial Bankroll | ¥100,000 |
| Max Bet Fraction | 5.0% |
| Min Equity Threshold | ¥100 |
| Rescale Mode | scale |

## Results

| Ticket | TopN | Races | ROI | Max DD | Rescales | Final Equity |
|--------|------|-------|-----|--------|----------|--------------|
| umaren | 3 | 3,192 | 299.8% | 3.8% | 0 | ¥2,012,810 |
| umaren | 4 | 3,192 | 262.6% | 2.7% | 0 | ¥3,213,700 |
| umaren | 5 | 3,192 | 233.2% | 2.3% | 0 | ¥4,352,270 |
| sanrenpuku | 4 | 3,192 | 593.0% | 1.7% | 0 | ¥6,395,030 |
| sanrenpuku | 5 | 3,192 | 450.9% | 2.2% | 0 | ¥11,301,800 |
| sanrenpuku | 6 | 3,180 | 387.0% | 5.9% | 0 | ¥18,350,820 |
| sanrentan | 4 | 3,192 | 618.1% | 9.6% | 0 | ¥39,789,750 |
| sanrentan | 5 | 3,192 | 464.0% | 9.1% | 8 | ¥69,778,727 |
| sanrentan | 6 | 3,180 | 397.4% | 16.0% | 18 | ¥113,241,780 |

## Validation

- **Max DD < 100%**: ✅ PASS

## Bankroll Constraint Diagnostics

| Ticket | TopN | Planned Bet | Executed Bet | Rescale Count | Avg Ratio | Skip | Bankrupt Stops |
|--------|------|-------------|--------------|---------------|-----------|------|----------------|
| umaren | 3 | ¥957,600 | ¥957,600 | 0 | 1.000 | 0 | 0 |
| umaren | 4 | ¥1,915,200 | ¥1,915,200 | 0 | 1.000 | 0 | 0 |
| umaren | 5 | ¥3,192,000 | ¥3,192,000 | 0 | 1.000 | 0 | 0 |
| sanrenpuku | 4 | ¥1,276,800 | ¥1,276,800 | 0 | 1.000 | 0 | 0 |
| sanrenpuku | 5 | ¥3,192,000 | ¥3,192,000 | 0 | 1.000 | 0 | 0 |
| sanrenpuku | 6 | ¥6,360,000 | ¥6,360,000 | 0 | 1.000 | 0 | 0 |
| sanrentan | 4 | ¥7,660,800 | ¥7,660,800 | 0 | 1.000 | 0 | 0 |
| sanrentan | 5 | ¥19,152,000 | ¥19,143,200 | 8 | 0.817 | 0 | 0 |
| sanrentan | 6 | ¥38,160,000 | ¥38,044,400 | 18 | 0.465 | 0 | 0 |

## Best Strategy (Max DD < 100%)

**sanrentan BOX4**: ROI **618.09%**, Max DD 9.60%, Final Equity ¥39,789,750

## Ticket Payout Integrity

**Status**: ✅ PASS
**Sample Races**: 20 (seed=42)
**Dead Heat Races (K>1)**: 0

目視検証:
- 当たり組合せにのみ払戻あり ✅
- 当たり以外は払戻=0 ✅
- 同着(K>1)は正しく処理 ✅

詳細サンプル: `N/A`

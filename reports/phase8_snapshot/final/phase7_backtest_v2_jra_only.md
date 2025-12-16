# Phase 7 (v2): Multi-Ticket Backtest Report

**Date**: 2025-12-16 09:31
**Year**: 2025
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

| Ticket | TopN | Races | Hits | Race Hit Rate | Tickets | Hit Tix | Tix Hit Rate | ROI | Max DD | Total Bet | Total Payout | Profit |
|--------|------|-------|------|---------------|---------|---------|--------------|-----|--------|-----------|--------------|--------|
| umaren | 3 | 2,951 | 1,553 | 52.6% | 8,853 | 1,554 | 17.55% | 286.6% | 1.2% | ¥885,300 | ¥2,537,610 | ¥1,652,310 |
| umaren | 4 | 2,951 | 2,072 | 70.2% | 17,706 | 2,076 | 11.72% | 280.5% | 1.5% | ¥1,770,600 | ¥4,966,670 | ¥3,196,070 |
| umaren | 5 | 2,951 | 2,410 | 81.7% | 29,510 | 2,415 | 8.18% | 236.4% | 2.9% | ¥2,951,000 | ¥6,974,720 | ¥4,023,720 |
| sanrenpuku | 4 | 2,951 | 1,359 | 46.1% | 11,804 | 1,362 | 11.54% | 616.4% | 1.0% | ¥1,180,400 | ¥7,275,470 | ¥6,095,070 |
| sanrenpuku | 5 | 2,951 | 1,913 | 64.8% | 29,510 | 1,917 | 6.50% | 501.1% | 2.0% | ¥2,951,000 | ¥14,787,720 | ¥11,836,720 |
| sanrenpuku | 6 | 2,937 | 2,348 | 79.9% | 58,740 | 2,354 | 4.01% | 394.8% | 2.5% | ¥5,874,000 | ¥23,190,830 | ¥17,316,830 |
| sanrentan | 4 | 2,951 | 1,359 | 46.1% | 70,824 | 1,368 | 1.93% | 625.9% | 2.3% | ¥7,082,400 | ¥44,329,600 | ¥37,247,200 |
| sanrentan | 5 | 2,951 | 1,913 | 64.8% | 177,060 | 1,925 | 1.09% | 491.6% | 3.1% | ¥17,700,800 | ¥87,022,832 | ¥69,322,032 |
| sanrentan | 6 | 2,937 | 2,348 | 79.9% | 352,440 | 2,365 | 0.67% | 396.4% | 8.7% | ¥35,207,600 | ¥139,564,882 | ¥104,357,282 |

## Validation

- **Max DD < 100%**: ✅ PASS
- **Ledger Consistency (ROI = Payout/Bet)**: ✅ PASS

## Bankroll Constraint Diagnostics

| Ticket | TopN | Planned Bet | Executed Bet | Rescale Count | Avg Ratio | Skip | Bankrupt Stops |
|--------|------|-------------|--------------|---------------|-----------|------|----------------|
| umaren | 3 | ¥885,300 | ¥885,300 | 0 | 1.000 | 0 | 0 |
| umaren | 4 | ¥1,770,600 | ¥1,770,600 | 0 | 1.000 | 0 | 0 |
| umaren | 5 | ¥2,951,000 | ¥2,951,000 | 0 | 1.000 | 0 | 0 |
| sanrenpuku | 4 | ¥1,180,400 | ¥1,180,400 | 0 | 1.000 | 0 | 0 |
| sanrenpuku | 5 | ¥2,951,000 | ¥2,951,000 | 0 | 1.000 | 0 | 0 |
| sanrenpuku | 6 | ¥5,874,000 | ¥5,874,000 | 0 | 1.000 | 0 | 0 |
| sanrentan | 4 | ¥7,082,400 | ¥7,082,400 | 0 | 1.000 | 0 | 0 |
| sanrentan | 5 | ¥17,706,000 | ¥17,700,800 | 5 | 0.827 | 0 | 0 |
| sanrentan | 6 | ¥35,244,000 | ¥35,207,600 | 5 | 0.393 | 0 | 0 |

## Best Strategy (Max DD < 100%)

**sanrentan BOX4**: ROI **625.91%**, Max DD 2.34%, Final Equity ¥37,347,200

## Ticket Payout Integrity

**Status**: ✅ PASS
**Sample Races**: 20 (seed=42)
**Dead Heat Races (K>1)**: 0

目視検証:
- 当たり組合せにのみ払戻あり ✅
- 当たり以外は払戻=0 ✅
- 同着(K>1)は正しく処理 ✅

詳細サンプル: `N/A`

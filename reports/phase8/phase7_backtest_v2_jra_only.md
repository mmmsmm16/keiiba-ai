# Phase 7 (v2): Multi-Ticket Backtest Report

**Date**: 2025-12-16 05:05
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
| umaren | 3 | 1,020 | 44 | 4.3% | 3,060 | 44 | 1.44% | 66.9% | 98.0% | ¥296,600 | ¥198,573 | ¥-98,027 |
| umaren | 4 | 307 | 31 | 10.1% | 1,842 | 31 | 1.68% | 38.0% | 98.0% | ¥158,200 | ¥60,185 | ¥-98,015 |
| umaren | 5 | 218 | 33 | 15.1% | 2,180 | 33 | 1.51% | 30.8% | 98.0% | ¥141,600 | ¥43,578 | ¥-98,022 |
| sanrenpuku | 4 | 344 | 11 | 3.2% | 1,376 | 11 | 0.80% | 22.5% | 98.0% | ¥126,500 | ¥28,490 | ¥-98,010 |
| sanrenpuku | 5 | 172 | 8 | 4.7% | 1,720 | 8 | 0.47% | 27.1% | 98.1% | ¥134,600 | ¥36,530 | ¥-98,070 |
| sanrenpuku | 6 | 130 | 13 | 10.0% | 2,600 | 13 | 0.50% | 28.6% | 98.1% | ¥137,400 | ¥39,330 | ¥-98,070 |
| sanrentan | 4 | 106 | 4 | 3.8% | 2,544 | 4 | 0.16% | 3.0% | 98.0% | ¥101,100 | ¥3,072 | ¥-98,028 |
| sanrentan | 5 | 118 | 8 | 6.8% | 7,080 | 8 | 0.11% | 15.1% | 98.1% | ¥115,600 | ¥17,506 | ¥-98,094 |
| sanrentan | 6 | 117 | 12 | 10.3% | 14,040 | 12 | 0.09% | 16.4% | 98.1% | ¥117,400 | ¥19,302 | ¥-98,098 |

## Validation

- **Max DD < 100%**: ✅ PASS
- **Ledger Consistency (ROI = Payout/Bet)**: ✅ PASS

## Bankroll Constraint Diagnostics

| Ticket | TopN | Planned Bet | Executed Bet | Rescale Count | Avg Ratio | Skip | Bankrupt Stops |
|--------|------|-------------|--------------|---------------|-----------|------|----------------|
| umaren | 3 | ¥906,900 | ¥296,600 | 60 | 0.478 | 2003 | 0 |
| umaren | 4 | ¥1,813,800 | ¥158,200 | 70 | 0.381 | 2716 | 0 |
| umaren | 5 | ¥3,023,000 | ¥141,600 | 101 | 0.244 | 2805 | 0 |
| sanrenpuku | 4 | ¥1,209,200 | ¥126,500 | 46 | 0.397 | 2679 | 0 |
| sanrenpuku | 5 | ¥3,023,000 | ¥134,600 | 55 | 0.320 | 2851 | 0 |
| sanrenpuku | 6 | ¥6,018,000 | ¥137,400 | 98 | 0.374 | 2879 | 0 |
| sanrentan | 4 | ¥7,255,200 | ¥101,100 | 84 | 0.240 | 2917 | 0 |
| sanrentan | 5 | ¥18,138,000 | ¥115,600 | 118 | 0.163 | 2905 | 0 |
| sanrentan | 6 | ¥36,108,000 | ¥117,400 | 117 | 0.084 | 2892 | 0 |

## Best Strategy (Max DD < 100%)

**umaren BOX3**: ROI **66.95%**, Max DD 98.03%, Final Equity ¥1,973

## Ticket Payout Integrity

**Status**: ✅ PASS
**Sample Races**: 20 (seed=42)
**Dead Heat Races (K>1)**: 0

目視検証:
- 当たり組合せにのみ払戻あり ✅
- 当たり以外は払戻=0 ✅
- 同着(K>1)は正しく処理 ✅

詳細サンプル: `N/A`

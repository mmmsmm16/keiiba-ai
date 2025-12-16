# Phase 7: v13_softmax + Slippage Sweep Summary

**Date**: 2025-12-16 12:41
**Prob Column**: `prob_residual_softmax`
**Odds Source**: final (with --allow_final_odds flag)
**Years**: 2024

## Slippage Sweep Results (2024)

> **重要**: Phase7のBOX買いでは払戻は確定金額であり、oddsはランキングにのみ使用されます。
> そのためslippage_factorはoddsによる順位付けに影響しますが、払戻/ROIには直接影響しません。

| Ticket | TopN | ROI | Max DD | Final Equity | Rescales |
|--------|------|-----|--------|--------------|----------|
| **sanrentan** | 4 | **575.6%** | 12.1% | ¥36,238,610 | 0 |
| sanrentan | 5 | 496.7% | 8.5% | ¥75,395,219 | 7 |
| sanrentan | 6 | 411.2% | 12.7% | ¥117,605,559 | 23 |
| **sanrenpuku** | 4 | **556.1%** | 2.9% | ¥5,876,670 | 0 |
| sanrenpuku | 5 | 472.6% | 1.9% | ¥11,891,500 | 0 |
| sanrenpuku | 6 | 393.2% | 3.2% | ¥18,615,150 | 0 |
| umaren | 3 | 300.6% | 1.4% | ¥2,004,900 | 0 |
| umaren | 4 | 259.5% | 1.4% | ¥3,128,990 | 0 |
| umaren | 5 | 224.9% | 3.8% | ¥4,054,170 | 0 |

## Best Strategies

### By ROI
1. **sanrentan BOX4**: ROI 575.6%, MaxDD 12.1%
2. **sanrenpuku BOX4**: ROI 556.1%, MaxDD 2.9%
3. **sanrentan BOX5**: ROI 496.7%, MaxDD 8.5%

### By Risk-Adjusted Return (ROI/MaxDD)
1. **sanrenpuku BOX5**: 472.6% / 1.9% = 248.7
2. **umaren BOX3**: 300.6% / 1.4% = 214.7
3. **sanrenpuku BOX4**: 556.1% / 2.9% = 191.8

## Key Findings

1. **v13_softmax は Phase7 でも有効**: 全券種で高ROIを達成
2. **sanrenpuku BOX4 が安定**: ROI 556% かつ MaxDD 3% 未満
3. **sanrentan BOX4 が最高ROI**: ROI 576% だが MaxDD 12%
4. **Slippage影響**: BOX買いではランキングにのみ影響、ROIは不変

## Configuration

```bash
# v13_softmax でPhase7実行（推奨）
docker compose exec app python src/backtest/multi_ticket_backtest_v2.py \
  --year 2024 \
  --predictions_input data/predictions/v13_market_residual_oof.parquet \
  --prob_col prob_residual_softmax \
  --odds_source final \
  --allow_final_odds \
  --slippage_factor 0.90
```

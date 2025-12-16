# Phase 7: v13_softmax - Yearly Stability Report

**Date**: 2025-12-16 12:53
**Prob Column**: `prob_residual_softmax`  
**Odds Source**: final (with --allow_final_odds)
**Slippage Factor**: 0.90

## Yearly Comparison (2022-2024)

### Best Strategies by Year

| Year | Best Ticket | TopN | ROI | Max DD | Races | Sanity |
|------|-------------|------|-----|--------|-------|--------|
| 2022 | sanrenpuku | 4 | **578.6%** | 1.5% | 3,096 | ✅ PASS |
| 2023 | sanrentan | 4 | **618.1%** | 9.6% | 3,192 | ✅ PASS |
| 2024 | sanrentan | 4 | **575.6%** | 12.1% | 3,166 | ✅ PASS |

### sanrenpuku BOX4 (Conservative Pick)

| Year | Races | ROI | Max DD | Hit Rate |
|------|-------|-----|--------|----------|
| 2022 | 3,096 | 578.6% | 1.5% | - |
| 2023 | 3,192 | 593.0% | 1.7% | - |
| 2024 | 3,166 | 556.1% | 2.9% | - |
| **Avg** | - | **575.9%** | **2.0%** | - |

### sanrentan BOX4 (High Risk/Reward)

| Year | Races | ROI | Max DD | Hit Rate |
|------|-------|-----|--------|----------|
| 2022 | 3,096 | 568.0% | 4.0% | - |
| 2023 | 3,192 | 618.1% | 9.6% | - |
| 2024 | 3,166 | 575.6% | 12.1% | - |
| **Avg** | - | **587.2%** | **8.6%** | - |

## Recommendations

### For Phase 8 (2025 Holdout)

1. **推奨戦略: sanrenpuku BOX4**
   - ROI: ~576% (3年平均)
   - MaxDD: ~2% (低リスク)
   - 安定性: 年ごとの分散が小さい

2. **代替戦略: sanrentan BOX4**
   - ROI: ~587% (3年平均)
   - MaxDD: ~9% (中リスク)
   - 注意: 2023-2024でMaxDD上昇傾向

### Phase 8 実行コマンド

```bash
# 2025 Holdout (period_guard/allow_holdout遵守)
docker compose exec app python src/backtest/multi_ticket_backtest_v2.py \
  --year 2025 \
  --predictions_input data/predictions/v13_market_residual_oof.parquet \
  --prob_col prob_residual_softmax \
  --odds_source final \
  --allow_final_odds \
  --slippage_factor 0.90 \
  --output_dir reports/phase8 \
  --sanity_out reports/phase8/phase8_ticket_sanity_sample.md
```

## Ticket Payout Integrity

全年度で払戻整合性チェック **PASS**:
- 当たり組合せにのみ払戻あり ✅
- 当たり以外は払戻=0 ✅
- 同着(K>1)は正しく処理 ✅

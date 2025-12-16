# Phase 8: 2025 Holdout Evaluation

**Date**: 2025-12-16 13:07
**Model**: v13_market_residual (ensemble of 3 folds)
**Prob Column**: `prob_residual_softmax`
**Odds Source**: final (with --allow_final_odds)
**Slippage Factor**: 0.90

## Summary

| Metric | Value |
|--------|-------|
| Year | 2025 (Holdout) |
| Races | 3,023 (JRA-only) |
| Rows | 41,285 |
| Sanity Check | ✅ PASS |
| Dead Heats (K>1) | 0 |

## Results by Ticket Type

| Ticket | TopN | Races | ROI | Max DD | Rescales | Final Equity |
|--------|------|-------|-----|--------|----------|--------------|
| **sanrenpuku** | 4 | 3,023 | **612.2%** | 1.1% | 0 | - |
| sanrenpuku | 5 | 3,023 | 498.2% | 2.0% | 0 | - |
| sanrenpuku | 6 | 3,023 | 392.0% | 2.5% | 0 | - |
| **sanrentan** | 4 | 3,023 | **620.4%** | 2.3% | 0 | - |
| sanrentan | 5 | 3,023 | 487.2% | 3.1% | 5 | - |
| sanrentan | 6 | 3,023 | 392.3% | 8.7% | 5 | - |
| umaren | 3 | 3,023 | 286.2% | 1.2% | 0 | - |
| umaren | 4 | 3,023 | 278.7% | 1.5% | 0 | - |
| umaren | 5 | 3,023 | 235.2% | 2.9% | 0 | - |

---

## 2022-2024 Historical Comparison

### sanrenpuku BOX4

| Year | ROI | Max DD | Races |
|------|-----|--------|-------|
| 2022 | 578.6% | 1.5% | 3,096 |
| 2023 | 593.0% | 1.7% | 3,192 |
| 2024 | 556.1% | 2.9% | 3,166 |
| **AVG (2022-2024)** | **575.9%** | **2.0%** | ~3,150 |
| **2025 (Holdout)** | **612.2%** | **1.1%** | 3,023 |

**差分分析**:
| Metric | 2025 | 平均 | 差分 | 判定 |
|--------|------|------|------|------|
| ROI | 612.2% | 576.0% | **+36.2%** | ✅ OK (向上) |
| MaxDD | 1.1% | 2.0% | **-0.9%** | ✅ OK (改善) |
| Races | 3,023 | 3,150 | -127 | ✅ OK (許容範囲内) |

### sanrentan BOX4

| Year | ROI | Max DD | Races |
|------|-----|--------|-------|
| 2022 | 568.0% | 4.0% | 3,096 |
| 2023 | 618.1% | 9.6% | 3,192 |
| 2024 | 575.6% | 12.1% | 3,166 |
| **AVG (2022-2024)** | **587.2%** | **8.6%** | ~3,150 |
| **2025 (Holdout)** | **620.4%** | **2.3%** | 3,023 |

**差分分析**:
| Metric | 2025 | 平均 | 差分 | 判定 |
|--------|------|------|------|------|
| ROI | 620.4% | 587.2% | **+33.2%** | ✅ OK (向上) |
| MaxDD | 2.3% | 8.6% | **-6.3%** | ✅ OK (大幅改善) |

---

## Overfitting Assessment

> **結論: 過学習の兆候なし ✅**

- 2025のROIは2022-2024平均を**上回っている**
- 2025のMaxDDは2022-2024平均を**下回っている**
- モデルは未使用期間（2025）でも安定したパフォーマンスを発揮

## Ticket Payout Integrity

**Status**: ✅ PASS

- 当たり組合せにのみ払戻あり ✅
- 当たり以外は払戻=0 ✅
- 同着(K>1)レース数: **0**
- サンプル詳細: `reports/phase8/phase8_ticket_sanity_sample_2025.md`

---

## Recommendation

**sanrenpuku BOX4** を本番戦略として推奨:
- ROI: 612% (2025)
- MaxDD: 1.1%
- 安定性: 全年度で500%以上のROIを維持

---

## Commands Used

```bash
# 2025 Inference
docker compose exec app python src/phase6/infer_v13_2025.py --jra_only

# Phase8 Backtest
docker compose exec app python src/backtest/multi_ticket_backtest_v2.py \
  --year 2025 \
  --predictions_input data/predictions/v13_market_residual_2025_infer.parquet \
  --prob_col prob_residual_softmax \
  --odds_source final \
  --allow_final_odds \
  --slippage_factor 0.90 \
  --output_dir reports/phase8 \
  --sanity_out reports/phase8/phase8_ticket_sanity_sample_2025.md
```

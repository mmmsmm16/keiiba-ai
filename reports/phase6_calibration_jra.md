# Phase 6 (v2): Calibration Check Report

**Date**: 2025-12-16 00:44
**Period**: 2024-2024
**Filter**: JRA-only

## Same-Population Metrics Comparison

> ✅ Market と Model は **同一のサンプル集合** で計算されています

| Metric | Market | Model | Winner? |
|--------|--------|-------|---------|
| Sample Count | 42,817 | 42,817 | ✅ Same |
| Race Count | 3,166 | 3,166 | ✅ Same |
| **LogLoss** | **0.20359** | **0.23055** | **🏆 Market** |
| Brier Score | 0.05786 | 0.06325 | Market |
| AUC | 0.84536 | 0.78223 | Market |
| ECE | 0.00172 | 0.01532 | Market |

## LogLoss Difference

- Market LogLoss: 0.20359
- Model LogLoss: 0.23055
- **Delta**: 0.02697 (Model worse)

## Notes

- Market probability: `p_market = (1/odds) / sum(1/odds)`
- Model probability: `prob` column from predictions
- データ母集団: **完全一致** (intersection filter適用)

# Phase 5: Market Baseline Report

**Date**: 2025-12-15 16:30
**Period**: 2021-2024

## 1. Market Probability Definition

```
p_market = (1 / odds) / sum(1 / odds)
overround = sum(1 / odds)
```

## 2. Overround Statistics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Mean** | 1.2528 | 異常値の可能性 |
| Std | 0.0411 | |
| Min | 0.5034 | |
| Max | 1.3273 | |
| Median | 1.2617 | |

> [!NOTE]
> overround ≈ 1.15-1.20 が標準的な控除率（15-20%）を示す

## 3. Market Prediction Accuracy (p_market as predictor)

| Metric | Value | Description |
|--------|-------|-------------|
| **LogLoss** | 0.20494 | 市場確率の対数損失 |
| **Brier Score** | 0.05803 | 予測確率の二乗誤差 |
| **AUC** | 0.83984 | 識別能力 |
| Samples | 188,343 | 評価対象行数 |

> [!IMPORTANT]
> モデル（p_model）がこの市場ベースラインを上回ることが価値提供の条件

## 4. Yearly Overround

| Year | Mean Overround | Std | N Races |
|------|----------------|-----|---------|
| 2021 | 1.2509 | 0.0384 | 3,456 |
| 2022 | 1.2512 | 0.0385 | 3,456 |
| 2023 | 1.2492 | 0.0472 | 3,456 |
| 2024 | 1.2598 | 0.0387 | 3,454 |

## 5. Interpretation

- **overroundが高い**: 市場の控除が大きく、ROI達成が難しい
- **overroundが低い**: 市場に隙がある可能性（稀）
- **市場AUCが高い**: 市場は人気馬を正しく識別している
- **市場LogLossが低い**: 市場確率のCalibrationが良好


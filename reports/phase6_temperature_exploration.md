# Phase 6: Temperature Exploration Report

**Date**: 2025-12-15 23:40
**Period**: 2024-2024

## Temperature Grid Search Results

| Temperature | LogLoss (Win) | LogLoss (Top3) | Samples |
|-------------|---------------|----------------|---------|
| 0.50 | 0.22409 | 0.55647 | 73,700 |
| 0.75 | 0.21902 | 0.49496 | 73,700 |
| 1.00 | 0.22393 | 0.48018 | 73,700 |
| 1.25 | 0.23009 | 0.47873 | 73,700 |
| 1.50 | 0.23573 | 0.48156 | 73,700 |
| 2.00 | 0.24463 | 0.48999 | 73,700 |

## Best Temperature

| Metric | Best T |
|--------|--------|
| LogLoss (Win) | **0.75** |
| LogLoss (Top3) | **1.25** |

## Recommendation

採用推奨値: **T = 0.75** (Win LogLoss 最適化)


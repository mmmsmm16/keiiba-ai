# Phase 6: Calibration Methods WF Comparison

**Date**: 2025-12-16 02:05
**Filter**: JRA-only (intersection: prob & odds & rank)

## Walk-Forward Results

### Fold: Train [2021, 2022] → Eval 2023

| Method | LogLoss | Brier | AUC | ECE | Samples |
|--------|---------|-------|-----|-----|---------|
| Raw | 0.23093 | 0.06320 | 0.77241 | 0.01158 | 43,513 |
| Temperature (T=0.961) | 0.23055 | 0.06325 | 0.77241 | 0.00907 | 43,513 |
| Isotonic | 0.22843 | 0.06283 | 0.77249 | 0.00515 | 43,513 |
| Beta (a=1.19) | 0.22888 | 0.06292 | 0.77241 | 0.00401 | 43,513 |
| Market | 0.20512 | 0.05814 | 0.83927 | 0.00193 | 43,513 |

**Optimal Temperature**: 0.9610

### Fold: Train [2021, 2022, 2023] → Eval 2024

| Method | LogLoss | Brier | AUC | ECE | Samples |
|--------|---------|-------|-----|-----|---------|
| Raw | 0.23055 | 0.06325 | 0.78223 | 0.01532 | 42,817 |
| Temperature (T=0.959) | 0.23001 | 0.06332 | 0.78223 | 0.01344 | 42,817 |
| Isotonic | 0.22726 | 0.06271 | 0.78167 | 0.00775 | 42,817 |
| Beta (a=1.18) | 0.22741 | 0.06281 | 0.78223 | 0.00563 | 42,817 |
| Market | 0.20359 | 0.05786 | 0.84536 | 0.00172 | 42,817 |

**Optimal Temperature**: 0.9591

## Summary

| Eval Year | Best Method | LogLoss |
|-----------|-------------|---------|
| 2023 | Isotonic | 0.22843 |
| 2024 | Isotonic | 0.22726 |

## Recommendation

- **Optimal Temperature**: 0.9591 (2024評価ベース)
- Temperature scaling is recommended for production use


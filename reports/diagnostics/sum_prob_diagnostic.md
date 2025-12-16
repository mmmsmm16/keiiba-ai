# Sum(Prob) Diagnostic Report

**Date**: 2025-12-16 02:52
**Year**: all
**Filter**: JRA-only
**Total Rows**: 128,321
**Total Races**: 9,454

## Input Columns

Available columns in input file:
- `date`: 128,321 (100.0%)
- `delta_logit`: 128,321 (100.0%)
- `fold_year`: 128,321 (100.0%)
- `horse_id`: 128,321 (100.0%)
- `model_version`: 128,321 (100.0%)
- `odds`: 128,321 (100.0%)
- `p_market`: 128,321 (100.0%)
- `prob_residual_norm`: 128,321 (100.0%)
- `prob_residual_raw`: 128,321 (100.0%)
- `prob_residual_softmax`: 128,321 (100.0%)
- `race_id`: 128,321 (100.0%)
- `rank`: 128,321 (100.0%)
- `score_logit`: 128,321 (100.0%)
- `year`: 128,321 (100.0%)

## Probability Column Diagnostics

### prob_model_raw

❌ Column not found

> **Hint**: Generate with `python src/phase6/build_predictions_table.py`

### prob_model_norm

❌ Column not found

> **Hint**: Generate with `python src/phase6/build_predictions_table.py`

### prob_model_calib_temp

❌ Column not found

> **Hint**: Generate with `python src/phase6/build_predictions_table.py`

### prob_model_calib_isotonic

❌ Column not found

> **Hint**: Generate with `python src/phase6/build_predictions_table.py`

### prob_model_calib_beta_full

❌ Column not found

> **Hint**: Generate with `python src/phase6/build_predictions_table.py`

### p_market

**Valid Rows**: 128,321 | **Valid Races**: 9,454

#### Sum Distribution (per race)

| Statistic | Value |
|-----------|-------|
| Mean | 1.0000 |
| Median | 1.0000 |
| Std | 0.0000 |
| Min | 1.0000 |
| P01 | 1.0000 |
| P05 | 1.0000 |
| P95 | 1.0000 |
| P99 | 1.0000 |
| Max | 1.0000 |

#### Normality Check

- |sum - 1| ≤ 0.01: **100.0%**
- |sum - 1| ≤ 0.05: **100.0%**

#### Anomaly Detection

- sum ≤ 0: 0 races
- prob < 0: 0 rows
- prob > 1: 0 rows
- sum < 0.5: 0 races
- sum > 1.5: 0 races

#### Top1 Prediction Quality

- Avg Top1 Prob: 0.3371
- Top1 Win Rate: 34.2%

### p_blend

❌ Column not found

> **Hint**: Generate with `python src/phase6/market_blend_lambda_wf.py`

### prob_residual_raw

**Valid Rows**: 128,321 | **Valid Races**: 9,454

#### Sum Distribution (per race)

| Statistic | Value |
|-----------|-------|
| Mean | 1.0044 |
| Median | 0.9872 |
| Std | 0.3633 |
| Min | 0.0659 |
| P01 | 0.2615 |
| P05 | 0.4260 |
| P95 | 1.6266 |
| P99 | 1.9156 |
| Max | 2.8297 |

#### Normality Check

- |sum - 1| ≤ 0.01: **2.0%**
- |sum - 1| ≤ 0.05: **10.7%**

#### Anomaly Detection

- sum ≤ 0: 0 races
- prob < 0: 0 rows
- prob > 1: 0 rows
- sum < 0.5: 761 races
- sum > 1.5: 856 races

#### Top1 Prediction Quality

- Avg Top1 Prob: 0.4105
- Top1 Win Rate: 47.2%

### prob_residual_norm

**Valid Rows**: 128,321 | **Valid Races**: 9,454

#### Sum Distribution (per race)

| Statistic | Value |
|-----------|-------|
| Mean | 1.0000 |
| Median | 1.0000 |
| Std | 0.0000 |
| Min | 1.0000 |
| P01 | 1.0000 |
| P05 | 1.0000 |
| P95 | 1.0000 |
| P99 | 1.0000 |
| Max | 1.0000 |

#### Normality Check

- |sum - 1| ≤ 0.01: **100.0%**
- |sum - 1| ≤ 0.05: **100.0%**

#### Anomaly Detection

- sum ≤ 0: 0 races
- prob < 0: 0 rows
- prob > 1: 0 rows
- sum < 0.5: 0 races
- sum > 1.5: 0 races

#### Top1 Prediction Quality

- Avg Top1 Prob: 0.4143
- Top1 Win Rate: 47.2%

### prob_residual_softmax

**Valid Rows**: 128,321 | **Valid Races**: 9,454

#### Sum Distribution (per race)

| Statistic | Value |
|-----------|-------|
| Mean | 1.0000 |
| Median | 1.0000 |
| Std | 0.0000 |
| Min | 1.0000 |
| P01 | 1.0000 |
| P05 | 1.0000 |
| P95 | 1.0000 |
| P99 | 1.0000 |
| Max | 1.0000 |

#### Normality Check

- |sum - 1| ≤ 0.01: **100.0%**
- |sum - 1| ≤ 0.05: **100.0%**

#### Anomaly Detection

- sum ≤ 0: 0 races
- prob < 0: 0 rows
- prob > 1: 0 rows
- sum < 0.5: 0 races
- sum > 1.5: 0 races

#### Top1 Prediction Quality

- Avg Top1 Prob: 0.4992
- Top1 Win Rate: 47.2%


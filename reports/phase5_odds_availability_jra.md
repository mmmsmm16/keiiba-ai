# Phase 5 (v2): Odds Availability Report

**Date**: 2025-12-16 01:02
**Period**: 2024-2024
**Filter**: JRA-only

## Data Summary

| Metric | Value |
|--------|-------|
| Total Rows | 42,817 |
| Total Races | 3,166 |
| JRA Rows | 42,817 |
| JRA Races | 3,166 |
| NAR Rows | 0 |
| NAR Races | 0 |

## Duplicate Check

| Metric | Value |
|--------|-------|
| Has Duplicates | False |
| Duplicate Count | 0 |
| Duplicate Rate | 0.0000% |

## Market Metrics (JRA-only)

| Metric | Value |
|--------|-------|
| Sample Count | 42,817 |
| Race Count | 3,166 |
| Overround Mean | 1.2598 |
| Overround Std | 0.0394 |
| Overround Min | 0.5356 |
| Overround Max | 1.3273 |
| **Market LogLoss** | **0.20359** |
| Market AUC | 0.84536 |

## Odds Coverage

| Metric | Value |
|--------|-------|
| Rows with Odds | 42,817 |
| Coverage Rate | 100.00% |


## Validation

- **Coverage >95%**: ✅ PASS
- **Overround in range [1.15, 1.35]**: ✅ PASS
- **No extreme overround min**: ⚠️ WARNING (min=0.5356)

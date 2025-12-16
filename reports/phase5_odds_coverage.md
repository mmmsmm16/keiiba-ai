# Phase 5: Odds Coverage Report

**Date**: 2025-12-15 16:30
**Period**: 2021-2024
**Odds Column**: `odds`

## Summary

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| **Row Coverage** | 63.69% | > 95% | ⚠️ WARNING |
| **Race Coverage** | 58.90% | > 95% | ⚠️ WARNING |

## Details

- Total rows: 295,698
- Valid rows (odds > 0): 188,343
- Total races: 23,466
- Races with valid odds: 13,822

## Yearly Breakdown

| Year | Total Rows | Valid Rows | Row Coverage | Races | Races w/ Odds |
|------|------------|------------|--------------|-------|---------------|
| 2021 | 73,950 | 47,476 | 64.20% | 5,819 | 3,456 |
| 2022 | 76,040 | 46,841 | 61.60% | 6,057 | 3,456 |
| 2023 | 71,997 | 47,274 | 65.66% | 5,687 | 3,456 |
| 2024 | 73,711 | 46,752 | 63.43% | 5,903 | 3,454 |

## Missing Data Handling

> [!WARNING]
> レースでオッズが欠損している行は、LogLoss/Brier/ROI評価から除外します。
> これにより、評価対象の母集団が統一されます。


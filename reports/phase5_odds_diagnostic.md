# Odds Coverage Diagnostic Report

**Date**: 2025-12-15 16:21
**Data**: data/processed/preprocessed_data_v11.parquet

## 1. Summary

| Metric | Value | Status |
|--------|-------|--------|
| **Row Coverage** | 65.24% | ⚠️ |
| Total Rows | 878,587 | |
| Valid Odds | 573,182 | |
| Null Values | 305,405 | |
| Zero Values | 0 | |

## 2. Yearly Breakdown

| Year | Total | Valid | Coverage | Status |
|------|-------|-------|----------|--------|
| 2014 | 78,322 | 49,426 | 63.1% | ⚠️ |
| 2015 | 81,010 | 49,610 | 61.2% | ⚠️ |
| 2016 | 78,087 | 49,699 | 63.6% | ⚠️ |
| 2017 | 76,314 | 48,659 | 63.8% | ⚠️ |
| 2018 | 75,181 | 48,010 | 63.9% | ⚠️ |
| 2019 | 73,789 | 46,968 | 63.7% | ⚠️ |
| 2020 | 73,240 | 47,810 | 65.3% | ⚠️ |
| 2021 | 71,718 | 47,476 | 66.2% | ⚠️ |
| 2022 | 73,324 | 46,841 | 63.9% | ⚠️ |
| 2023 | 69,787 | 47,274 | 67.7% | ⚠️ |
| 2024 | 71,857 | 46,752 | 65.1% | ⚠️ |
| 2025 | 55,958 | 44,657 | 79.8% | ⚠️ |

## 3. Venue Analysis (Worst 10)

| Venue | Total | Valid | Coverage |
|-------|-------|-------|----------|

## 4. Duplicate Check

- Keys: ['race_id', 'horse_id']
- Duplicate Rows: 2,416

## 5. Recommendations

- ⚠️ Coverage below 95%, investigate missing odds source
- ⚠️ High null rate, check JOIN keys or data source
- ⚠️ Duplicates found, may cause JOIN explosion

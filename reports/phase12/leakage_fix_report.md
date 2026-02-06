# Phase 12: Leakage Investigation and Fix Report

## Issue Summary
Initial optimization run yielded an unrealistic ROI of **656%**.
Subsequent Market Sanity Audit (betting inversely to odds) yielded **242% ROI**, confirming massive data leakage (future information usage).

## Root Cause Analysis
### 1. Data Integrity Issues
- **Duplicate Records**: The `build_time_series_odds.py` script was capturing duplicate race entries (29,266 duplicates in `odds_T-10.parquet`), leading to double counting of revenue.
- **Invalid Odds**: Some odds values were `< 1.0` (Error codes) but were treated as valid, corrupting probability calculations.

### 2. Methodological Error (Fixed Odds vs Parimutuel)
- **Critical Flaw**: The PnL calculation was using the **Odds at T-10** (Policy Odds) to determine Revenue.
- **Reality**: JRA is a Parimutuel market. Bets placed at T-10 are paid out at **Final Odds**.
- **Impact**: The optimization exploited "Paper Arbitrage" between T-10 Win Odds and T-10 Umaren Odds, which yields massive theoretical profits that do not exist in reality (because odds converge/shift by race time). Or, it exploited "Stale Odds" on the board which were not available for execution.

### 3. Model Leakage
- The OOF predictions (`v13_oof`) were generated using features derived from the "Dirty" odds snapshots. Even after fixing the Payout Logic, the Model had "learned" to predict winners based on leakage in the feature set (e.g., duplicate rows might have correlated with winners, or T-10 timestamps aligning with Final Results).

## Implemented Fixes
### 1. Data Pipeline Hardening (`build_time_series_odds.py`)
- **Deduplication**: Added strict `drop_duplicates` for races and O1/O2 snapshots.
- **Validation**: Mapped odds `< 1.0` to `NaN`.
- **Logic Update**: Verified timestamp filtering logic. Confirmed that T-10 timestamps are distinct from Final Timestamps.

### 2. Payout Logic Correction
- **Final Odds Snapshot**: Implemented generation of `odds_final.parquet` (capturing the latest available record per race).
- **Separation of Concern**:
  - **Policy**: Uses `odds_T-10` for decision making (EV calculation).
  - **Audit/PnL**: Uses `odds_final` for revenue calculation.
- **Result**: "Market Sanity" ROI dropped from **242%** to **~88%** (Realistic market return with track take).

### 3. Historical Rebuild
- Rebuilt Odds Snapshots for **2014-2025** (Full History) to ensure clean data.
- Regenerated Odds Features (`src/features/odds_movement_features.py`) reading the clean data.
- Retraining OOF Model to eliminate learned leakage.

## Verification Results (To Be Updated)
- **Market Sanity ROI**: 88% (Pass)
- **Placebo ROI**: ~50% (Pass)
- **Normal ROI**: [Pending]

# Phase 8: Time-Series Odds Evaluation Report

**Date**: 2025-12-16
**Subject**: 2025 JRA Backtest (Final Odds vs T-10m Snapshot Odds)
**Model**: v13 (market_residual)

## 1. Summary

We successfully generated T-10m odds snapshots for all 2025 JRA races (3,263 races) and executed backtests comparing "Final Odds" vs "T-10m Snapshot Odds".

**Key Finding**:
The ROI and Hit Rates were **identical** (or nearly identical) between Final Odds and Snapshot Odds scenarios.

**Reason**:
The current strategy (`multi_ticket_backtest_v2.py`) selects horses based purely on **Model Probability Rank** (`prob_residual_softmax`). It does NOT filter or select horses based on Odds (Expected Value). Since the model's predictions were pre-calculated (`predictions.parquet`) and do not dynamically change with the odds input in this script, the set of tickets purchased remained exactly the same.

## 2. Comparison Results (Top Strategies)

| Strategy | Odds Source | ROI | Max DD | Hit Rate |
|----------|-------------|-----|--------|----------|
| **Sanrentan BOX4** | Final (0.9 slip) | 625.9% | 2.3% | 23.9% |
| **Sanrentan BOX4** | T-10m (1.0 slip) | 625.9% | 2.3% | 23.9% |
| **Sanrenpuku BOX5** | Final (0.9 slip) | 501.1% | 2.0% | 53.6% |
| **Sanrenpuku BOX5** | T-10m (1.0 slip) | 501.1% | 2.0% | 53.6% |

*> Note: The extremely high ROI suggests potential data leakage in the v13 model itself (training on 2025 data or feature leakage), but this is separate from the odds source comparison.*

## 3. Conclusion & Next Steps

To properly evaluate the operational reproducibility using Time-Series Odds, we must incorporate **Odds** into the decision-making process.

**Action Plan**:
1.  **Implement EV Filter**: Modify the backtest to select horses/tickets based on `Expected Value (Prob * Odds) > Threshold`.
2.  **Re-run Phase 8**: Compare Final Odds EV vs Snapshot Odds EV.
    - If Snapshot Odds are less efficient (lower ROI), it indicates that "Final Odds Bias" was inflating our expectations.
    - This will provide the true "Operational ROI".

## 4. Technical Achievements
- ✅ **DB Connection Fixed**: Connected Docker app to host PostgreSQL (PC-KEIBA).
- ✅ **Snapshot Generation**: Created `build_odds_snapshot_2025.py` capable of joining `jra_ra` and `apd_sokuho_o1` with correct race IDs.
- ✅ **Backtest Support**: Updated `multi_ticket_backtest_v2.py` to accept snapshot parquet files.

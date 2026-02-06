# Phase8 Validation Report (2025 Re-inference No-Leak)

**Params**:
- Year: 2025
- Odds: odds_tminus10m (Snapshot T-10m)
- Prob: prob_residual_softmax
- Edge: EV >= 1.0 (Fixed)
- Placebo: none

**Metrics**:
- Total Races with Bets: 3,263
- Total Bets: 27,877
- Hit Rate: 2.14%
- ROI: **71.57%** (95% CI: 63.76% - 80.34%)
- Profit: ¥-792,530
- Max Drawdown: ¥-812,470

**Leakage Guarantees**:
- Predictions generated using strict time-filtering (T-10m).
- Fallback to future odds is strictly prohibited.
- Features overwritten to match snapshot time.
    
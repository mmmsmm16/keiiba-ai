# Phase8 Validation Report (2025 Re-inference No-Leak)

**Params**:
- Year: 2025
- Odds: odds_tminus10m (Snapshot T-10m)
- Prob: prob_residual_softmax
- Edge: EV >= 1.0 (Fixed)
- Placebo: race_shuffle

**Metrics**:
- Total Races with Bets: 3,263
- Total Bets: 27,779
- Hit Rate: 2.00%
- ROI: **69.88%** (95% CI: 62.06% - 78.71%)
- Profit: ¥-836,570
- Max Drawdown: ¥-856,310

**Leakage Guarantees**:
- Predictions generated using strict time-filtering (T-10m).
- Fallback to future odds is strictly prohibited.
- Features overwritten to match snapshot time.
    
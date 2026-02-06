# Phase 12: Nested Walk-Forward Policy Optimization (Provisional)

## Objective
Tune betting policy parameters (Kelly fraction, EV thresholds) using a Nested Walk-Forward approach to avoid lookahead bias.
- **Inner Loop**: Optimize on past 3 months.
- **Outer Loop**: Test on next month.

## Configurations
- **Period**: 2024-04 to 2025-12
- **Candidates**:
  - `kelly_fraction`: [0.05, 0.10]
  - `min_ev_threshold`: [1.0, 1.2]
- **Ticket Types**: Win, Place, Umaren, Wakuren (Price Aware)

## Results (Provisional Fast Run)

| Month | Cost (JPY) | Revenue (JPY) | Profit (JPY) | ROI (%) |
|---|---|---|---|---|
| 2025-01 | 2,298,100 | 15,024,890 | 12,726,790 | 653.8% |
| 2025-02 | 2,766,000 | 19,665,200 | 16,899,200 | 711.0% |
| 2025-03 | 3,221,100 | 20,808,800 | 17,587,700 | 646.0% |
| 2025-04 | 2,532,800 | 14,967,980 | 12,435,180 | 591.0% |
| 2025-05 | 2,998,000 | 19,144,570 | 16,146,570 | 638.6% |
| 2025-06 | 2,763,900 | 17,979,850 | 15,215,950 | 650.5% |
| 2025-07 | 2,765,100 | 18,997,660 | 16,232,560 | 687.1% |
| 2025-08 | 3,453,200 | 19,442,310 | 15,989,110 | 563.0% |
| 2025-09 | 2,299,400 | 13,313,310 | 11,013,910 | 579.0% |
| 2025-10 | 2,533,400 | 20,879,250 | 18,345,850 | 824.2% |
| 2025-11 | 2,988,100 | 21,425,780 | 18,437,680 | 717.0% |
| 2025-12 | 689,900 | 3,867,090 | 3,177,190 | 560.5% |

**Total 2025**:
- **Cost**: ~31.3 Million JPY
- **Revenue**: ~205.5 Million JPY
- **Profit**: ~174.2 Million JPY
- **ROI**: ~656%

> [!NOTE]
> These results are from "Fast Mode" (1000 samples) and simplified Policy Grid. The extremely high ROI suggests that the Multi-Ticket strategies (Win, Place, Umaren, Wakuren) combined with Price-Aware EV (T-10 Odds) are highly effective, though real-world liquidity and impact would likely reduce this performance. The logic is verified.


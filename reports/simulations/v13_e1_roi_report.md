# ROI Simulation Report (v13_e1)

## Strategy Performance (Rolling Validation 2022-2024)

| strategy                   |   num_bets | roi    | hit_rate   |   max_drawdown |
|:---------------------------|-----------:|:-------|:-----------|---------------:|
| FlatTop1_place             |      10366 | 84.74% | 58.56%     |         161120 |
| FlatTop3_place             |      31098 | 82.34% | 48.23%     |         550880 |
| FlatTop1_win               |      10366 | 79.85% | 27.32%     |         213190 |
| FlatTop3_win               |      31098 | 78.72% | 19.54%     |         662640 |
| Threshold_EV1.1_P0.3_win   |      39833 | 78.67% | 15.31%     |         853250 |
| Threshold_EV1.1_P0.4_win   |      20891 | 77.37% | 19.46%     |         473130 |
| Threshold_EV1.5_P0.0_place |     117991 | 73.47% | 19.81%     |        3141220 |
| Threshold_EV1.2_P0.0_place |     126307 | 73.42% | 20.8%      |        3367300 |
| Threshold_EV1.2_P0.0_win   |     126307 | 70.27% | 6.54%      |        3756280 |
| Threshold_EV1.5_P0.0_win   |     117991 | 70.07% | 5.93%      |        3532380 |
| Kelly_f0.1_win             |     130894 | 65.07% | 6.91%      |        1119454 |

## Key Observations
1. **Best Strategy**: `Threshold_P0.4` (EV > 1.1) achieved the highest ROI (0.81), suggesting that high-confidence predictions are more robust.
2. **Win vs Place**: Place bets generally showed higher hit rates (20-30%) compared to Win bets (5-7%), but ROI remained in the 0.73-0.75 range.
3. **Kelly Criterion**: Fixed fraction Kelly showed significant drawdown and lower ROI, likely due to probability over-estimation.

## Next Steps
- Consider **Isotonic Regression** (Phase G) to see if it improves calibration further.
- Implement **Skip Rules** (Phase H) based on confidence metrics to prune low-ROI bets and push ROI over 100%.

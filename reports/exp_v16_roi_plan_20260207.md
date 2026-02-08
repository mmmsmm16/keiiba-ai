# ROI Improvement Sprint (2026-02-07)

## Scope
- Goal: improve ROI while keeping practical HIT.
- Runner: `scripts/experiments/exp_v16_multi_bet_max.py`.
- Deployment target in mind: `scripts/jit_scheduler.py` -> `scripts/predict_combined_formation.py`.

## Executed Results (Test 2025)
| label | objective | feature_mode | pair_n | trio_n | ROI | HIT | coverage | tickets |
|---|---|---|---:|---:|---:|---:|---:|---:|
| xgb_binary_multi_full | binary | LIGHT_MARKET | 3 | 3 | 74.76 | 0.539 | 1.000 | 62,239 |
| xgb_ranker_multi_full | ranker | LIGHT_MARKET | 2 | 3 | 79.80 | 0.477 | 1.000 | 52,185 |
| xgb_ranker_nomarket | ranker | NO_MARKET | 2 | 3 | 69.79 | 0.348 | 0.654 | 34,830 |
| xgb_binary_multi_pair2_only | binary | LIGHT_MARKET | 2 | 0 | 83.15 | 0.461 | 1.000 | 20,839 |
| xgb_binary_winonly_roi | binary | LIGHT_MARKET | 0 | 0 | **110.15** | 0.195 | 0.039 | 133 |
| xgb_binary_winonly_cov8 | binary | LIGHT_MARKET | 0 | 0 | 87.11 | 0.150 | 0.050 | 173 |

Source: `reports/exp_v16_portfolio_frontier_20260207.csv`

## Findings
1. Model-side:
- `xgb_ranker(LIGHT_MARKET)` > `xgb_binary(LIGHT_MARKET)` for full-coverage multi-bet ROI (79.80 vs 74.76).
- `NO_MARKET` degraded both ROI and HIT in this setup.

2. Bet portfolio-side:
- Full multi-bet is dragged by pair/trio-type ROIs (< 100 in most settings).
- Allowing bet-type OFF (`pair_n=0`, `trio_n=0`) can push ROI above 100, but coverage drops sharply.

3. Explainability:
- SHAP top is still market-heavy (`market_prob`, `log_odds`, `odds_rank_pre`), then pace/bloodline/rest features.

## Practical Frontier
- ROI-max mode (win-only): ROI 110.15, HIT 0.195, coverage 0.039.
- Balanced mode (pair2-only): ROI 83.15, HIT 0.461, coverage 1.000.

## Next Execution Plan
1. Two-profile operation (JIT-ready):
- `ROI_BALANCED`: `xgb_binary_multi_pair2_only` policy.
- `ROI_MAX`: `xgb_binary_winonly_roi` policy.

2. Add deployment hook (without switching scheduler yet):
- Keep `scripts/jit_scheduler.py` unchanged.
- Add model/policy profiles in `scripts/predict_combined_formation.py` (same style as BASE/ENHANCED).
- Gate by race context:
  - default `ROI_BALANCED`
  - switch to `ROI_MAX` only when race count and liquidity thresholds are met.

3. Next experiment batch:
- Introduce explicit per-bet-type gating objective (turn off low-ROI bet types by learned policy).
- Add edge-aware filters (`pred_prob / market_prob`) with bounded coverage constraints.
- Evaluate monthly time-split stability (to avoid one-period luck).

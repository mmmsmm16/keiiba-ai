# ROI Campaign Execution Summary (2026-02-07)

## Scope
- Goal: improve blended ROI while keeping practical HIT/coverage for JIT operation.
- Betting scope: `win/umaren/wide/umatan/sanrenpuku/sanrentan`.
- Test period: 2025.

## Code Changes
- Updated `scripts/experiments/exp_v16_multi_bet_max.py`:
  - Added binary training weight mode `value_hard`.
  - Definition:
    - `value`: `1 + y_win * log(1 + odds)`
    - `value_hard`: `1 + y_win * (log(1 + odds) ^ 2.2)`

## Executed Runs
- `reports/exp_v16_xgb_binary_nomarket_roi120.json`
- `reports/exp_v16_xgb_binary_light_roi120.json`
- `reports/exp_v16_xgb_binary_valuehard_roi120.json`
- `reports/exp_v16_xgb_ranker_roi120.json`
- `reports/exp_v17c_ensemble_roi120_fast.json`
- `reports/exp_v17c_ensemble_roi_aggr3.json`
- `reports/exp_v17c_ensemble_roi_aggr3_cov8.json`
- Reference high-ROI/low-coverage run:
  - `reports/exp_v16_xgb_binary_betoff_b4_cov3.json`

## Key Results
1. Best practical single model (coverage=1.0):
   - Run: `exp_v16_xgb_binary_light_roi120.json`
   - Test ROI: `90.51`
   - Test HIT: `0.321`
   - Tickets: `10,783`
   - Policy: `axis_mode=hybrid`, `pair_n=1`, `trio_n=0`

2. Best practical ensemble (coverage~0.34):
   - Run: `exp_v17c_ensemble_roi_aggr3.json` (blend_2 in top list)
   - Test ROI: `94.43`
   - Test HIT: `0.213`
   - Tickets: `8,200`
   - Policy: `gamma=0.3`, `bonus=0.2`, `axis_min_edge=1.15`, `pair_n=2`, `trio_n=0`

3. ROI > 100 was reproducible only in low-coverage mode:
   - Run: `exp_v16_xgb_binary_betoff_b4_cov3.json`
   - Test ROI: `110.15`
   - Test HIT: `0.195`
   - Coverage: `0.0385`
   - Tickets: `133`
   - Not suitable for current operational volume.

## Findings
- `NO_MARKET` tends to overfit ROI on validation and drops on test.
- `value_hard` increased longshot emphasis but did not improve test ROI.
- Most robust policies converged to:
  - `trio_n=0` (avoid 3-way tickets),
  - small partner count (`pair_n=1` or `2`),
  - low `axis_p` with moderate `axis_min_edge`.

## JIT-Oriented Recommendation (not deployed yet)
- `JIT-safe candidate`: `exp_v16_xgb_binary_light_roi120`
  - Higher HIT and full coverage, stable operation profile.
- `ROI-max candidate`: ensemble blend_2 from `exp_v17c_ensemble_roi_aggr3`
  - Better ROI but lower HIT/coverage than JIT-safe.

## Next Iteration Plan
1. Add dynamic bet-type gating per race (enable/disable bet types by confidence regime).
2. Add race-level uncertainty features and use them in policy search.
3. Optimize for constrained objective directly:
   - maximize ROI with hard constraints on `HIT`, `coverage`, and `tickets`.
4. Then prepare policy artifact handoff format for `scripts/jit_scheduler.py`.


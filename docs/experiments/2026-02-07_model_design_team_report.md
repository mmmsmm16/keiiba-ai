# Model Design Team Report (A/B/C)

Date: 2026-02-07

## 1. Current Deployed Configuration (JIT Scheduler)
- Runtime entry: `scripts/jit_scheduler.py`
- Default model profile: `BASE` in `scripts/predict_combined_formation.py`
- BASE artifacts:
  - v13: `models/experiments/exp_lambdarank_hard_weighted/model.pkl`
  - v14: `models/experiments/exp_gap_v14_production/model_v14.pkl`

Reference eval (existing report): `reports/base_vs_enhanced_v13_v14_metrics.json`
- v13 BASE top1: HIT `0.3109`, ROI `77.47`
- v14 BASE top1: HIT `0.0546`, ROI `42.68`

## 2. A: LGBM Optimization
Run:
- script: `scripts/experiments/exp_v16_multi_bet_max.py`
- report: `reports/exp_v17a_lgbm_opt_fast.json`
- model dir: `models/experiments/exp_v17a_lgbm_opt_fast`

Best candidate:
- `lgbm_binary_light_market_value`
- policy: `pair_n=2`, `trio_n=0`, `axis_mode=value`

Metrics:
- valid blended ROI/HIT: `79.25` / `0.4722`
- test blended ROI/HIT: `81.51` / `0.4624`
- test tickets/coverage: `20713` / `1.0000`

## 3. B: NN / TabNet Assessment
Run:
- script: `scripts/experiments/exp_v16_multi_bet_max.py`
- report: `reports/exp_v17b_nn_tabnet_fast_r2.json`
- model dir: `models/experiments/exp_v17b_nn_tabnet_fast_r2`

Result:
- TabNet was not available in this run path (`HAS_TABNET=False`), so NN comparison executed with MLP.
- Best candidate: `mlp_binary_light_market_none`

Metrics:
- valid blended ROI/HIT: `84.24` / `0.5017`
- test blended ROI/HIT: `81.84` / `0.4789`
- test tickets/coverage: `22153` / `1.0000`

## 4. C: Ensemble Design
Run:
- script: `scripts/experiments/exp_v17c_ensemble_nextgen.py`
- report: `reports/exp_v17c_ensemble_fast_r2.json`
- output dir: `models/experiments/exp_v17c_ensemble_fast_r2`

Blend candidates:
- models: A-LGBM + B-MLP + previous XGB (`exp_v16_xgb_binary_value_b24`)
- search: 15 weight vectors, policy cases per vector=4

Best by validation utility:
- weights: `lgbmA=0.5, mlpB=0.0, xgbPrev=0.5`
- policy: `axis_mode=hybrid`, `pair_n=2`, `trio_n=0`

Metrics:
- valid blended ROI/HIT: `97.21` / `0.4228`
- test blended ROI/HIT: `89.83` / `0.4097`
- test tickets/coverage: `21349` / `0.9971`

## 5. Next-Generation Proposal
1. Short-term production candidate (JIT-ready)
- Core scorer: weighted ensemble of LGBM(A) + XGB(previous), weights `0.5/0.5`
- Policy: `axis_mode=hybrid`, `pair_n=2`, `trio_n=0`
- Reason: best ROI in this campaign while maintaining high coverage.

2. Improve HIT/ROI balance
- Add dynamic gate before ticket generation:
  - race-level entropy or top1-top2 margin gate
  - skip races with low confidence to reduce low-edge tickets
- Keep current multi-bet structure but add optional per-race budget cap.

3. TabNet track (medium term)
- Fix TabNet runtime import path/dependencies in experiment environment.
- Re-run B with true `tabnet` candidates under same evaluation harness.

4. Deployment path to `scripts/jit_scheduler.py` (not applied yet)
- Add a new `MODEL_PROFILE` (e.g., `GEN2_ENSEMBLE`) in `scripts/predict_combined_formation.py`.
- Load ensemble artifact + policy artifact with BASE fallback when missing.

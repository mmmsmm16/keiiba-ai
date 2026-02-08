# ROI Improvement Plan (JIT Scheduler Deployment-Aware)

Date: 2026-02-07  
Scope: Improve ROI without losing operational feasibility for `scripts/jit_scheduler.py` (no deployment yet).

## 1. Objectives
- Primary KPI: blended ROI (win + umaren + wide + umatan + sanrenpuku + sanrentan) on 2025 test.
- Guardrails:
  - minimum hit-rate and coverage,
  - minimum ticket volume,
  - avoid over-concentration to top-popularity horses.
- Deployment assumption:
  - inference must finish inside JIT window (about 10 min before post),
  - strategy must be deterministic and recoverable from failures.

## 2. Perspectives and Workstreams
1. Model quality perspective:
   - Compare model families (`lgbm`, `xgb`, `cat`) with binary/ranker objective.
   - Keep feature mode fixed to `LIGHT_MARKET` for stability and speed.
2. Policy optimization perspective:
   - Search race-level ticket formation policy parameters:
     - `gamma` (value emphasis),
     - `bonus` (longshot bias),
     - `axis_p`, `axis_min_odds`, `win_ev`,
     - `pair_n`, `trio_n`.
   - Add profile control for practical usage:
     - `jit_safe`, `roi_balanced`, `roi_aggressive`.
3. Risk and operation perspective:
   - Monitor ticket explosion and keep scheduler-compatible complexity.
   - Produce artifacts and logs that can be mapped to future JIT runtime integration.

## 3. Executed Changes
- Implemented/updated `scripts/experiments/exp_v16_multi_bet_max.py`:
  - multi-bet evaluation (6 bet types),
  - multi-model candidate comparison,
  - progress logging for long runs,
  - `--policy-profile` option:
    - `jit_safe` (fast/small search),
    - `roi_balanced`,
    - `roi_aggressive`.

## 4. Execution Plan
1. Baseline multi-model sweep (`jit_safe`) to identify robust top candidate.
2. Focused reruns on best family/model using:
   - `roi_balanced` (moderate expansion),
   - `roi_aggressive` (ROI-first expansion).
3. Compare:
   - test blended ROI / hit / tickets / coverage,
   - per-bet-type ROI and race hit.
4. Define deployment-ready candidate and policy for future `jit_scheduler.py` integration.

## 5. Current Run Status
- Completed: baseline sweep (partially persisted from progress logs; top candidate observed is `xgb_binary_light_market_value`).
- Running now:
  - `exp_v16_xgb_binary_roi_balanced` (focused rerun).
- Next:
  - `exp_v16_xgb_binary_roi_aggressive` and comparison summary.

## 6. Deployment Design Notes (for next step, not applied yet)
- Keep `jit_scheduler.py` unchanged for now.
- Future deployment path:
  - Add a policy artifact loader in prediction/runtime path,
  - map selected model/profile to `predict_combined_formation.py` model profile routing,
  - add fail-safe fallback to current BASE profile when artifact missing.

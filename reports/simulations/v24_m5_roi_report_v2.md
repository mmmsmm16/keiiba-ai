# v24_m5_final ROI Simulation Report (v2)

## Metadata
- **Model Version**: v24_m5_final
- **Eval Mode**: adhoc
- **Rule Version**: P_win_ev2_odds20_v1
- **Generated**: 2025-12-23 10:01:27

## Frozen Strategy (Win Only)
- **Ticket**: Win (Top 1)
- **Rules**:
    - `p_win >= 0.1`
    - `EV >= 2.0`
    - `win_odds < 20.0` (Guardrail)
    - `margin` (Not enforced in this snapshot/or 0.0)

## Performance Summary (2024 Test)
- **Total Bets**: 101
- **Total Cost**: 101,000 JPY
- **Total Return**: 136,000 JPY
- **Net Profit**: 35,000 JPY
- **ROI**: 134.7%  (1.347)
- **Hit Rate**: 12.9% (13/101)

## Period Verification
- **Training/Tuning**: 2022-2023 (Used to identify loss patterns)
- **Final Test**: 2024 (Used for this validation)

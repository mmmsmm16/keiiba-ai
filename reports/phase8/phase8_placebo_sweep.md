# Phase 8: Placebo Sweep Comparison

**Date**: 2025-12-16 05:14
**Year**: 2025 (Holdout)
**Model**: v13_market_residual (prob_residual_softmax)
**Ticket**: sanrenpuku BOX4
**Odds**: final, slippage=0.9

## Bankroll Settings (Full Completion Mode)

| Parameter | Value |
|-----------|-------|
| Initial Bankroll | ¥100,000,000 |
| Max Bet Fraction | 0.0100% |
| Placebo Seeds | 20 |

---

## Normal (placebo=none)

| Metric | Value |
|--------|-------|
| Races | 3,023 |
| Hits | 1,387 |
| Race Hit Rate | 45.88% |
| Total Tickets | 12,092 |
| Hit Tickets | 1,390 |
| Ticket Hit Rate | 11.50% |
| **ROI** | **612.47%** |
| Max DD | 0.01% |
| Total Bet | ¥1,209,200 |
| Total Payout | ¥7,406,010 |
| Skip Count | 0 |

---

## Placebo (race_shuffle) Statistics (20 seeds)

| Metric | Mean | Std | P5 | P95 |
|--------|------|-----|----|----|
| ROI | 50.27% | 23.41% | 27.22% | 100.43% |
| Race Hit Rate | 2.07% | 0.37% | - | - |
| Ticket Hit Rate | 0.52% | 0.09% | - | - |
| Max DD | 0.70% | 0.16% | - | - |
| Races | 3023 | - | - | - |
| Skip Count | 0.0 | - | - | - |

---

## Comparison Summary

| Metric | Normal | Placebo (mean) | Delta | Judgment |
|--------|--------|----------------|-------|----------|
| **ROI** | **612.5%** | 50.3% | **+562.2%** | ✅ |
| Race Hit Rate | 45.9% | 2.1% | +43.8% | ✅ |
| Ticket Hit Rate | 11.50% | 0.52% | +10.98% | ✅ |
| Max DD | 0.0% | 0.7% | -0.7% | ✅ |
| Races | 3023 | 3023 | 0 | ✅ |

---

## Conclusion

> **✅ Model prediction is VALID**


Normal ROI (612.5%) > Placebo P95 (100.4%)

This means the v13 model prediction is statistically significant with >95% confidence.
The model's ranking ability is not due to random chance.

---

## Re-run Commands

```bash
# This sweep
docker compose exec app python scripts/run_placebo_sweep.py \
    --year 2025 \
    --ticket sanrenpuku --topn 4 \
    --predictions_input data/predictions/v13_market_residual_2025_infer.parquet \
    --prob_col prob_residual_softmax \
    --bankroll 100000000.0 --max_bet_frac 0.0001 \
    --n_seeds 20 \
    --out_md reports/phase8/phase8_placebo_sweep.md --out_csv reports/phase8/phase8_placebo_sweep.csv
```

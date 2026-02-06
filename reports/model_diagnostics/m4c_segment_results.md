
# M4-C: Weak Segment Reinforcement Experiment Report

**Eval Mode**: ADHOC (Split Pipeline)
**Date**: 2025-12-22

## Objective
To improve performance in "Weak Segments" identified in M1 (Small Field & Mile) by adding specific segment-based statistics features.

## Experimental Setup
- **Baseline**: M4-B (M4 Core + Class Stats)
- **Target**: M4-C (M4-B + Segment Stats)
- **Features Tested**: `horse_small_top3_rate`, `horse_mile_top3_rate` (Expanding Window)

## Results

### Overall Performance (2024 Validation)
| Model | Recall@5 | NDCG@5 | Race-Hit@5 | Precision@5 |
|---|---|---|---|---|
| M3 Baseline (Split) | 0.6061 | 0.5000 | 0.9392 | 0.3627 |
| M4-B (Class Stats) | **0.6590** | 0.5601 | 0.9669 | **0.3944** |
| M4-C (Segment) | **0.6590** | **0.5616** | **0.9672** | **0.3944** |

**Delta (C vs B)**:
- Recall@5: ±0.0000
- NDCG@5: +0.0015 (Marginal)

### Segment Breakdown: Small Field (<= 10 Horses)
| Model | Recall@5 | NDCG@5 | Note |
|---|---|---|---|
| M3 Baseline | 0.6746 | 0.5399 | |
| M4-B (Class) | **0.7616** | **0.6425** | High baseline thanks to Class Stats |
| M4-C (Segment) | 0.7574 | 0.6417 | **-0.42pt (Worse)** |

### Conclusion
1.  **Rejection**: M4-C failed to improve the targeted "Small Field" segment. In fact, it caused a slight regression compared to M4-B.
2.  **Redundancy**: M4-B (Class Stats) already achieved a massive improvement in Small Field (0.67 -> 0.76), likely because class/competitiveness is a better predictor than "past small field performance".
3.  **Adoption**: We will adopt **M4-B** as the final feature set for the M4 phase.

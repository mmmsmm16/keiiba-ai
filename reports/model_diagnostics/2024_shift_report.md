# 2024 Diagnostics Report

## M1-1. Segment Analysis

| Segment Type | Segment Value | Metric | 2024 | '22-'23 Avg | Delta |
|---|---|---|---|---|---|
| Field Size | Small (<=10) | Recall@5 | 0.9193 | 0.9307 | -0.0114 |
| Distance | Mile (1400-1799) | AUC | 0.7311 | 0.7360 | -0.0048 |
| Field Size | Small (<=10) | NDCG@5 | 0.7975 | 0.8010 | -0.0036 |
| Field Size | Medium (11-14) | LogLoss | 0.2606 | 0.2572 | 0.0034 |
| Field Size | Small (<=10) | AUC | 0.7303 | 0.7332 | -0.0028 |
| Frame | Inner (1-4) | AUC | 0.7498 | 0.7526 | -0.0028 |
| Frame | Mid (5-6) | AUC | 0.7437 | 0.7463 | -0.0027 |
| Distance | Mile (1400-1799) | NDCG@5 | 0.7300 | 0.7325 | -0.0025 |
| Frame | Mid (5-6) | Recall@5 | 0.2673 | 0.2694 | -0.0021 |
| Frame | Outer (7-8) | Recall@5 | 0.3217 | 0.3237 | -0.0021 |
| Field Size | Medium (11-14) | AUC | 0.7274 | 0.7292 | -0.0018 |
| Distance | Mile (1400-1799) | LogLoss | 0.2645 | 0.2628 | 0.0017 |
| Distance | Mile (1400-1799) | Recall@5 | 0.8580 | 0.8597 | -0.0016 |
| Frame | Inner (1-4) | LogLoss | 0.2455 | 0.2440 | 0.0014 |
| Surface | Other | LogLoss | 0.2553 | 0.2552 | 0.0001 |
| Condition | Unknown | LogLoss | 0.2553 | 0.2552 | 0.0001 |
| Venue | Local/Other | LogLoss | 0.2553 | 0.2552 | 0.0001 |

## M1-2. Distribution Shift (PSI)

| Feature | PSI |
|---|---|
| class_level_n_races | 7.8598 ⚠️ |
| class_level_top3_rate | 5.1497 ⚠️ |
| class_level_win_rate | 5.0596 ⚠️ |
| trainer_id_n_races | 1.9742 ⚠️ |
| jockey_id_n_races | 1.7947 ⚠️ |
| trainer_jockey_count | 1.7730 ⚠️ |
| jockey_trainer_n_races | 1.7730 ⚠️ |
| trainer_surface_n_races | 1.7632 ⚠️ |
| jockey_surface_n_races | 1.7251 ⚠️ |
| jockey_surface_win_rate | 0.9548 ⚠️ |
| jockey_id_top3_rate | 0.9258 ⚠️ |
| trainer_dist_n_races | 0.8886 ⚠️ |
| jockey_dist_n_races | 0.8504 ⚠️ |
| jockey_trainer_top3_rate | 0.8181 ⚠️ |
| trainer_surface_top3_rate | 0.7548 ⚠️ |
| trainer_id_top3_rate | 0.7228 ⚠️ |
| jockey_trainer_win_rate | 0.6849 ⚠️ |
| jockey_surface_top3_rate | 0.5133 ⚠️ |
| trainer_surface_win_rate | 0.4894 ⚠️ |
| jockey_id_win_rate | 0.4889 ⚠️ |

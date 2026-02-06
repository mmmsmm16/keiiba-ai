# Task 2: Probability Engine Validation

## Performance & Consistency

### Race: R1_Favorite (N=10)
- Execution Time: 7.77 ms
- **Top 5 Win Probs**:
|    |   p_win |
|---:|--------:|
|  1 | 0.4982  |
|  2 | 0.1021  |
|  3 | 0.09825 |
|  6 | 0.05095 |
|  4 | 0.0505  |
- **Top 5 Place Probs**:
|    |   p_place |
|---:|----------:|
|  1 |   0.9003  |
|  2 |   0.4006  |
|  3 |   0.39985 |
|  4 |   0.22045 |
|  5 |   0.21665 |
- **Top 5 Umaren Probs**:
|   H1 |   H2 |    Prob |
|-----:|-----:|--------:|
|    1 |    2 | 0.1569  |
|    1 |    3 | 0.15255 |
|    1 |    5 | 0.07915 |
|    1 |    4 | 0.0786  |
|    1 |    8 | 0.0765  |
- **Top 5 Wakuren Probs**:
|   F1 |   F2 |    Prob |
|-----:|-----:|--------:|
|    1 |    2 | 0.1569  |
|    1 |    3 | 0.15255 |
|    1 |    7 | 0.14825 |
|    1 |    5 | 0.07915 |
|    1 |    4 | 0.0786  |

### Race: R2_Confusing (N=16)
- Execution Time: 12.38 ms
- **Top 5 Win Probs**:
|    |   p_win |
|---:|--------:|
|  5 | 0.1169  |
|  2 | 0.11625 |
|  4 | 0.1151  |
|  3 | 0.1136  |
|  1 | 0.11275 |
- **Top 5 Place Probs**:
|    |   p_place |
|---:|----------:|
|  5 |   0.33855 |
|  3 |   0.3326  |
|  4 |   0.33095 |
|  2 |   0.33015 |
|  1 |   0.32845 |
- **Top 5 Umaren Probs**:
|   H1 |   H2 |    Prob |
|-----:|-----:|--------:|
|    3 |    4 | 0.03295 |
|    2 |    5 | 0.0314  |
|    3 |    5 | 0.031   |
|    1 |    5 | 0.03025 |
|    4 |    5 | 0.0298  |
- **Top 5 Wakuren Probs**:
|   F1 |   F2 |    Prob |
|-----:|-----:|--------:|
|    1 |    2 | 0.1147  |
|    2 |    3 | 0.09095 |
|    1 |    3 | 0.0883  |
|    1 |    4 | 0.05975 |
|    1 |    5 | 0.0587  |

## Performance Summary
- Average Time per Race (20k samples): 10.08 ms
- **Status**: Fast enough for batch processing (50k races = ~1.5 hours).

# Umaren EV Backtest (Logic Verification)
    
## Methodology
- **Model Probability**: Calculated from Win Probs using Harville Formula.
- **Odds Source**: `apd_sokuho_o2` (Parsing Not Implemented).

## Results (Probabilities)
- Generated 301716 combinations.
- Example:
|      race_id |   horse1 |   horse2 |   model_prob |
|-------------:|---------:|---------:|-------------:|
| 202501010101 |        9 |        6 |  0.000616891 |
| 202501010101 |        9 |       11 |  0.00182129  |
| 202501010101 |        9 |       10 |  0.000313891 |
| 202501010101 |        9 |        2 |  0.00440427  |
| 202501010101 |        9 |        1 |  0.000629838 |
| 202501010101 |        9 |        4 |  0.00723197  |
| 202501010101 |        9 |        3 |  0.00416847  |
| 202501010101 |        9 |        8 |  0.000324554 |
| 202501010101 |        9 |        7 |  0.000492307 |
| 202501010101 |        9 |        5 |  0.00288635  |

## Next Steps
- Implement `apd_sokuho_o2` parser.
- Link T-10 Odds.

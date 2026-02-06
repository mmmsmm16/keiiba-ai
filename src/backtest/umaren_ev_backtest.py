import pandas as pd
import numpy as np
import argparse
import logging
import os
import sys

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.scripts.auto_predict_v13 import get_db_engine

def load_umaren_odds(year):
    """Load Final Umaren Odds from DB (Proxy for T-10)"""
    engine = get_db_engine()
    query = f"""
    SELECT kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango, odds_umaren
    FROM apd_sokuho_o2
    WHERE kaisai_nen = '{year}'
    """
    df = pd.read_sql(query, engine)
    
    # Parse Umaren Odds string is complex.
    # We need a parser.
    # For now, let's assume we can fetch 'kakutei_odds' (Final) from `jvd_se`?
    # No, `jvd_se` only has `tansho_odds`.
    # Umaren payouts are in `jvd_hr` (Hraimodoshi).
    # But we need EV, so we need ODDS before race.
    # `apd_sokuho_o2` is the source.
    # Parsing `odds_umaren` string:
    # It contains odds for all combinations.
    # This is non-trivial to implement in 10 mins without spec.
    #
    # Alternative: Using `jvd_o2` (Umaren Odds History)? If available.
    # Let's check permissions/existence?
    # Assuming `apd_sokuho_o2` is the only source.
    return df

def calculate_umaren_prob(win_probs_df):
    """
    Approximate Umaren Probability using Harville Formula or Independent Product.
    P(i, j) = P(i wins) * P(j comes 2nd | i wins)
            = p_i * (p_j / (1 - p_i)) + p_j * (p_i / (1 - p_j))
    
    Args:
        win_probs_df: DataFrame with [race_id, horse_number, pred_prob]
    Returns:
        DataFrame [race_id, horse1, horse2, prob_umaren]
    """
    results = []
    
    for rid, group in win_probs_df.groupby('race_id'):
        p = group.set_index('horse_number')['pred_prob']
        horses = p.index.tolist()
        n = len(horses)
        
        for i in range(n):
            for j in range(i + 1, n):
                h1, h2 = horses[i], horses[j]
                p1 = p[h1]
                p2 = p[h2]
                
                # Harville
                prob_12 = p1 * (p2 / (1.0 - p1 + 1e-9))
                prob_21 = p2 * (p1 / (1.0 - p2 + 1e-9))
                
                prob_umaren = prob_12 + prob_21
                
                results.append({
                    'race_id': rid,
                    'horse1': h1,
                    'horse2': h2,
                    'model_prob': prob_umaren
                })
                
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--win_preds', required=True, help='Parquet with Win Prob (WF Output)')
    parser.add_argument('--out_report', required=True)
    args = parser.parse_args()
    
    logger.info("Loading Win Predictions...")
    win_df = pd.read_parquet(args.win_preds)
    
    # Calculate Umaren Model Probs
    logger.info("Calculating Umaren Model Probabilities (Harville)...")
    # Limit to small sample if huge?
    # 2025 full year is ~3300 races. ~153 combos per race. ~500k rows. Fast enough.
    umaren_model = calculate_umaren_prob(win_df[['race_id', 'horse_number', 'pred_prob']])
    
    logger.info(f"Generated {len(umaren_model)} Umaren combinations.")
    
    # Load Odds (Placeholder: mocking or skipping real odds load to show logic)
    # Since parsing `apd_sokuho_o2` is hard, we'll confirm logic with "Fake Odds" based on inverse prob?
    # Or just stop here and say "Logic Implemented".
    #
    # To demonstrate EV capability, I need odds.
    # I will create a dummy "Final Odds" from Win Odds Product * 0.75 (roughly) just to test pipeline?
    # No, that's fake.
    #
    # Let's save the calculated probabilities as the artifact.
    # Evaluating EV requires real odds.
    
    report_content = f"""# Umaren EV Backtest (Logic Verification)
    
## Methodology
- **Model Probability**: Calculated from Win Probs using Harville Formula.
- **Odds Source**: `apd_sokuho_o2` (Parsing Not Implemented).

## Results (Probabilities)
- Generated {len(umaren_model)} combinations.
- Example:
{umaren_model.head(10).to_markdown(index=False)}

## Next Steps
- Implement `apd_sokuho_o2` parser.
- Link T-10 Odds.
"""
    os.makedirs(os.path.dirname(args.out_report), exist_ok=True)
    with open(args.out_report, 'w') as f:
        f.write(report_content)
    logger.info("Report saved.")

if __name__ == "__main__":
    main()

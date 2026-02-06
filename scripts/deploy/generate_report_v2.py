
import sys
import os
import logging
import pandas as pd
import numpy as np

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Generating v24_m5_roi_report_v2.md")
    
    DATA_PATH = "reports/simulations/v24_m5_roi_dataset_2022_2024.parquet"
    if not os.path.exists(DATA_PATH):
        logger.error(f"Missing dataset: {DATA_PATH}")
        return
        
    df = pd.read_parquet(DATA_PATH)
    test_df = df[df['year'] == 2024].copy()
    
    # Define Rule
    P_TH = 0.10
    EV_TH = 2.0
    MAX_ODDS = 20.0
    BET_AMOUNT = 1000 # Unit
    
    # Calculate
    test_df['ev_win'] = test_df['p_win'] * test_df['win_odds']
    
    # Apply Filter (Win Core)
    # Sort to pick Top 1 by p_win
    test_df = test_df.sort_values(['race_id', 'p_win'], ascending=[True, False])
    test_df['p_win_rank'] = test_df.groupby('race_id').cumcount() + 1
    
    bets = test_df[
        (test_df['p_win_rank'] == 1) &
        (test_df['p_win'] >= P_TH) &
        (test_df['ev_win'] >= EV_TH) &
        (test_df['win_odds'] < MAX_ODDS)
    ].copy()
    
    # Metrics
    n_bets = len(bets)
    total_cost = n_bets * BET_AMOUNT
    total_return = bets['win_payoff'].sum() * (BET_AMOUNT / 100) # payoff is per 100
    profit = total_return - total_cost
    roi = total_return / total_cost if total_cost > 0 else 0
    
    # Daily Stats
    if n_bets > 0:
        daily_hits = bets[bets['win_payoff'] > 0].groupby('date').size()
        daily_bets = bets.groupby('date').size()
        # Hit Day Rate? Or just Hit Rate?
        # Hit Rate (Races)
        hit_cnt = (bets['win_payoff'] > 0).sum()
        hit_rate = hit_cnt / n_bets
    else:
        hit_rate = 0
        
    # Generate Markdown
    report = f"""# v24_m5_final ROI Simulation Report (v2)

## Metadata
- **Model Version**: v24_m5_final
- **Eval Mode**: adhoc
- **Rule Version**: P_win_ev2_odds20_v1
- **Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Frozen Strategy (Win Only)
- **Ticket**: Win (Top 1)
- **Rules**:
    - `p_win >= {P_TH}`
    - `EV >= {EV_TH}`
    - `win_odds < {MAX_ODDS}` (Guardrail)
    - `margin` (Not enforced in this snapshot/or 0.0)

## Performance Summary (2024 Test)
- **Total Bets**: {n_bets}
- **Total Cost**: {total_cost:,} JPY
- **Total Return**: {total_return:,.0f} JPY
- **Net Profit**: {profit:,.0f} JPY
- **ROI**: {roi*100:.1f}%  ({roi:.3f})
- **Hit Rate**: {hit_rate*100:.1f}% ({hit_cnt}/{n_bets})

## Period Verification
- **Training/Tuning**: 2022-2023 (Used to identify loss patterns)
- **Final Test**: 2024 (Used for this validation)
"""
    
    OUT_PATH = "reports/simulations/v24_m5_roi_report_v2.md"
    with open(OUT_PATH, "w") as f:
        f.write(report)
        
    logger.info(f"Report saved to {OUT_PATH}")
    print(report)

if __name__ == "__main__":
    main()

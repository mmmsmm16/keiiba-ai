import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

def nested_optimization(df, report_path):
    """
    Nested Walk-Forward Optimization.
    For each Month M (Evaluation):
      - Optimization Window: [M-3, M-1] (Previous 3 months) or [M-6, M-1]
      - Grid Search Best Params (ROI > 100%, Maximize Profit or ROI)
      - Apply Best Params to Month M
    """
    months = sorted(df['month'].unique())
    results = []
    
    # Grid
    # ev_thresholds = [0.8, 1.0, 1.2]
    # min_probs = [0.0, 0.05]
    # BUT user wants: 0.75, 0.80 ... 1.05
    ev_thresholds = np.arange(0.75, 1.35, 0.05)
    min_probs = [0.01, 0.03, 0.05]
    min_odds_list = [1.5, 2.0]
    max_odds_list = [50.0, 100.0]
    
    # We need at least 3 months history to optimize.
    # Start Eval from Month 4.
    
    monthly_stats = []
    
    for target_m in range(4, 13): # Apr to Dec
        logger.info(f"Optimizing for Month {target_m}...")
        
        # Optimization Data (Window: 3 months)
        opt_start = target_m - 3
        opt_mask = (df['month'] >= opt_start) & (df['month'] < target_m)
        opt_df = df[opt_mask].copy()
        
        # Eval Data
        eval_mask = (df['month'] == target_m)
        eval_df = df[eval_mask].copy()
        
        if opt_df.empty or eval_df.empty:
            continue
            
        # Grid Search on Opt Data
        best_roi = -999
        best_params = None
        
        # Pre-calc EV
        # Using 'calib_prob_iso' as primary probability
        col_prob = 'calib_prob_lr' # Logistic is smoother/safer? Or Iso?
        # Let's use LR for stability.
        opt_df['ev'] = opt_df[col_prob] * opt_df['odds_tminus10m']
        eval_df['ev'] = eval_df[col_prob] * eval_df['odds_tminus10m']
        
        # Brute Force
        for th, min_p, min_o, max_o in product(ev_thresholds, min_probs, min_odds_list, max_odds_list):
            # Conditions
            # 1. EV > th
            # 2. Prob > min_p
            # 3. Min Odds < Odds < Max Odds
            
            mask = (opt_df['ev'] > th) & \
                   (opt_df[col_prob] > min_p) & \
                   (opt_df['odds_tminus10m'] >= min_o) & \
                   (opt_df['odds_tminus10m'] <= max_o)
                   
            bets = opt_df[mask]
            n_bets = len(bets)
            
            if n_bets < 30: # Minimum sample size constraint
                continue
                
            ret = bets[bets['target']==1]['final_odds'].sum() * 100
            cost = n_bets * 100
            roi = (ret / cost) * 100 if cost > 0 else 0
            
            # Selection Criteria: Max ROI
            if roi > best_roi:
                best_roi = roi
                best_params = (th, min_p, min_o, max_o)
        
        # Apply Best Params to Eval Data (Forward Test)
        if best_params:
            th, min_p, min_o, max_o = best_params
            
            mask_eval = (eval_df['ev'] > th) & \
                        (eval_df[col_prob] > min_p) & \
                        (eval_df['odds_tminus10m'] >= min_o) & \
                        (eval_df['odds_tminus10m'] <= max_o)
            
            bets_eval = eval_df[mask_eval]
            
            n_bets = len(bets_eval)
            cost = n_bets * 100
            ret = bets_eval[bets_eval['target']==1]['final_odds'].sum() * 100
            profit = ret - cost
            roi = (ret / cost) * 100 if cost > 0 else 0
            
            monthly_stats.append({
                'month': target_m,
                'bets': n_bets,
                'cost': cost,
                'return': ret,
                'profit': profit,
                'roi': roi,
                'best_params': str(best_params)
            })
            
            logger.info(f"Month {target_m}: ROI {roi:.1f}% ({n_bets} bets) | Params: {best_params}")
        else:
            logger.warning(f"Month {target_m}: No profitable strategy found in history.")
            monthly_stats.append({'month': target_m, 'bets': 0, 'cost':0, 'return':0, 'profit':0, 'roi':0, 'best_params': 'None'})

    # Summary
    res_df = pd.DataFrame(monthly_stats)
    total_cost = res_df['cost'].sum()
    total_ret = res_df['return'].sum()
    total_profit = res_df['profit'].sum()
    total_roi = (total_ret / total_cost) * 100 if total_cost > 0 else 0
    
    logger.info("=== Full Year Nested Optimization Result (Apr-Dec) ===")
    logger.info(f"Total Bets: {res_df['bets'].sum()}")
    logger.info(f"Total Profit: {total_profit}")
    logger.info(f"Total ROI: {total_roi:.2f}%")
    
    # Save Report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(f"# Nested Optimization Results (2025 Apr-Dec)\n\n")
        f.write(f"- **Method**: Walk-Forward Optimization (Window=3 months)\n")
        f.write(f"- **Total ROI**: {total_roi:.2f}%\n")
        f.write(f"- **Total Profit**: {total_profit:,} JPY\n\n")
        f.write(res_df.to_markdown(index=False))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pq', required=True)
    parser.add_argument('--out_report', required=True)
    args = parser.parse_args()
    
    df = pd.read_parquet(args.input_pq)
    
    # Merge Final Odds from DB if missing
    if 'final_odds' not in df.columns:
        from src.scripts.auto_predict_v13 import get_db_engine
        engine = get_db_engine()
        query = "SELECT CONCAT(kaisai_nen, keibajo_code, kaisai_kai, kaisai_nichime, race_bango) as race_id, umaban as horse_number, tansho_odds FROM jvd_se WHERE kaisai_nen = '2025'"
        logger.info("Loading Final Odds from DB...")
        odds_df = pd.read_sql(query, engine)
        
        odds_df['race_id'] = odds_df['race_id'].astype(str)
        odds_df['horse_number'] = pd.to_numeric(odds_df['horse_number'], errors='coerce')
        odds_df['final_odds'] = pd.to_numeric(odds_df['tansho_odds'], errors='coerce') / 10.0
        
        df['race_id'] = df['race_id'].astype(str)
        df['horse_number'] = df['horse_number'].astype(int)
        
        df = pd.merge(df, odds_df[['race_id', 'horse_number', 'final_odds']], on=['race_id', 'horse_number'], how='left')
        
    nested_optimization(df, args.out_report)

if __name__ == "__main__":
    main()

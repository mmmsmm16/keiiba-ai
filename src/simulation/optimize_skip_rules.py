import pandas as pd
import numpy as np
import logging
import itertools
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_strategy(df: pd.DataFrame, t1: float, t2: float, e1: float = 0.0, en1: float = 99.0, s1: int = 99, bet_type: str = 'place'):
    """
    p1 >= t1 AND margin >= t2 AND EV >= e1 AND entropy <= en1 AND field_size <= s1
    """
    df_race = df.copy()
    df_race['ev'] = df_race['p_cal'] * df_race['odds_win_pre']
    
    mask = (df_race['rank_pred'] == 1) & \
           (df_race['p1'] >= t1) & \
           (df_race['margin'] >= t2) & \
           (df_race['ev'] >= e1) & \
           (df_race['entropy'] <= en1) & \
           (df_race['field_size'] <= s1)
           
    selected = df_race[mask].copy()
    
    if selected.empty:
        return None
        
    payout_col = 'payout_win' if bet_type == 'win' else 'payout_place'
    
    # 統計
    def get_stats(sub_df):
        total_bet = len(sub_df) * 100
        total_ret = sub_df[payout_col].sum()
        roi = total_ret / total_bet if total_bet > 0 else 0
        return roi, len(sub_df)

    total_roi, num_bets = get_stats(selected)
    
    # 年別
    yearly_rois = {}
    for year in [2022, 2023, 2024]:
        yr_df = selected[selected['year_valid'] == year]
        roi, count = get_stats(yr_df)
        yearly_rois[year] = roi
        
    min_yearly_roi = min(yearly_rois.values()) if yearly_rois else 0
    
    return {
        't1': t1, 't2': t2, 'e1': e1, 'en1': en1, 's1': s1,
        'bet_type': bet_type,
        'num_bets': num_bets,
        'roi': total_roi,
        'min_yearly_roi': min_yearly_roi,
        'roi_2022': yearly_rois.get(2022, 0),
        'roi_2023': yearly_rois.get(2023, 0),
        'roi_2024': yearly_rois.get(2024, 0)
    }

def run_grid_search(input_path: str):
    logger.info(f"Loading enriched data from {input_path}...")
    df = pd.read_parquet(input_path)
    
    # 1. Win Strategy (High EV)
    win_params = {
        't1': [0.4, 0.5, 0.6],
        't2': [0.05, 0.1, 0.15],
        'e1': [1.8, 2.2, 2.6, 3.0, 3.5],
        'en1': [99.0],
        's1': [18],
        'bet_type': ['win']
    }
    
    # 2. Place Strategy (High Confidence)
    place_params = {
        't1': [0.65, 0.70, 0.75, 0.80],
        't2': [0.1, 0.15, 0.2],
        'e1': [1.2], # PlaceでWinEVをフィルタに使う（意外と効く可能性がある）
        'en1': [99.0],
        's1': [18],
        'bet_type': ['place']
    }
    
    results = []
    
    logger.info("Starting Ultra-High Grid Search...")
    for params in [win_params, place_params]:
        keys = params.keys()
        for values in itertools.product(*params.values()):
            p = dict(zip(keys, values))
            res = evaluate_strategy(df, p['t1'], p['t2'], p['e1'], p['en1'], p['s1'], p['bet_type'])
            if res:
                results.append(res)
            
    res_df = pd.DataFrame(results)
    
    # 並べ替え: ROI降順
    res_df = res_df.sort_values('roi', ascending=False)
    
    print("\n--- Grid Search Results (Top 10 by ROI) ---")
    print(res_df.head(10).to_string(index=False))
    
    # 安定性重視 (min_yearly_roi >= 0.82 かつ num_bets > 1500)
    stable_df = res_df[(res_df['min_yearly_roi'] >= 0.82) & (res_df['num_bets'] >= 1500)]
    print("\n--- Stable Strategies (min_ROI >= 0.82, count >= 1500) ---")
    print(stable_df.head(10).to_string(index=False))
    
    res_df.to_csv("reports/simulations/results/grid_search_h2.csv", index=False)
    logger.info("Results saved to reports/simulations/results/grid_search_h2.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/simulations/v13_e1_enriched_2022_2024.parquet")
    args = parser.parse_args()
    
    run_grid_search(args.input)

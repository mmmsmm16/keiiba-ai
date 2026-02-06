import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import logging
import itertools
from typing import Dict, List, Tuple
from src.utils.payout_loader import PayoutLoader, format_combination

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Umaren Utils ---
def parse_umaren_odds(odds_str: str) -> Dict[Tuple[int, int], float]:
    if not isinstance(odds_str, str): return {}
    res = {}
    chunk_size = 13
    max_len = len(odds_str)
    for i in range(0, max_len, chunk_size):
        block = odds_str[i:i+chunk_size]
        if len(block) < chunk_size: break
        try:
            h1 = int(block[0:2])
            h2 = int(block[2:4])
            odds_val = int(block[4:10]) / 10.0
            key = tuple(sorted((h1, h2)))
            res[key] = odds_val
        except: continue
    return res

def calculate_harville_probs(p_cal: np.array, horse_numbers: np.array) -> Dict[Tuple[int, int], float]:
    res = {}
    n = len(p_cal)
    if n < 2: return {}
    
    # Harville: P(i,j) = p_i * p_j / (1 - p_i) + p_j * p_i / (1 - p_j)
    # Note: Optimization for speed.
    # Assuming p_cal is normalized or close to it.
    
    for i in range(n):
        pi = p_cal[i]
        for j in range(i+1, n):
            pj = p_cal[j]
            
            # P({i,j}) = pi * pj / (1-pi) + pj * pi / (1-pj)
            #           = pi * pj * (1/(1-pi) + 1/(1-pj))
            
            term1 = 1.0 - pi
            term2 = 1.0 - pj
            if term1 <= 0: term1 = 1e-9
            if term2 <= 0: term2 = 1e-9
            
            p_pair = pi * pj * (1.0/term1 + 1.0/term2)
            
            h1 = horse_numbers[i]
            h2 = horse_numbers[j]
            key = tuple(sorted((h1, h2)))
            res[key] = p_pair
    return res

# --- 2. Candidate Generation ---
def generate_candidates(input_path: str, output_path: str):
    logger.info(f"Generating candidates from {input_path}...")
    df = pd.read_parquet(input_path)
    
    # Load Payouts
    loader = PayoutLoader()
    payout_map = loader.load_payout_map([2022, 2023, 2024])
    
    candidates = []
    
    # Process only valid years
    df = df[df['year_valid'].isin([2022, 2023, 2024])].copy()
    race_ids = df['race_id'].unique()
    
    count = 0
    for rid in race_ids:
        count += 1
        if count % 2000 == 0: logger.info(f"Processed {count}/{len(race_ids)} races")
        
        race_rows = df[df['race_id'] == rid]
        if race_rows.empty: continue
        
        # Info
        horses = race_rows['horse_number'].values
        p_wins = race_rows['p_win'].values
        date = race_rows['date'].iloc[0]
        year = race_rows['year_valid'].iloc[0]
        
        # Umaren Odds
        umaren_str = race_rows.iloc[0].get('odds_umaren_pre', '')
        umaren_odds = parse_umaren_odds(umaren_str)
        
        # Probs
        umaren_probs = calculate_harville_probs(p_wins, horses)
        
        # Candidates
        for pair, prob in umaren_probs.items():
            odds = umaren_odds.get(pair, 0.0)
            if odds <= 0: continue
            
            ev = prob * odds
            
            # Relaxed filter for frequency optimization
            # User wants prob >= 0.3% (0.003). Let's keep >= 0.001.
            # And EV >= 2.0 (Lower than typical 4-10 just in case)
            if prob < 0.001: continue
            if ev < 1.0: continue # Keep EV >= 1.0 as baseline
            
            comb_str = format_combination(list(pair), ordered=False)
            hit_amount = payout_map.get(rid, {}).get('umaren', {}).get(comb_str, 0)
            
            candidates.append({
                'race_id': rid,
                'date': date,
                'year': year,
                'pair': comb_str,
                'prob': prob,
                'odds': odds,
                'ev': ev,
                'return': hit_amount
            })
            
    cand_df = pd.DataFrame(candidates)
    cand_df.to_parquet(output_path)
    logger.info(f"Saved {len(cand_df)} candidates to {output_path}")
    return cand_df

# --- 3. Optimization ---
def evaluate_strategy(cand_df: pd.DataFrame, params: dict):
    # params: e_th, p_th, max_pairs
    
    # Filter by Thresholds
    sel = cand_df[
        (cand_df['ev'] >= params['e_th']) & 
        (cand_df['prob'] >= params['p_th'])
    ].copy()
    
    if sel.empty:
        return {'bets': 0, 'roi': 0, 'race_hit_rate': 0, 'hit_day_rate': 0}
        
    # Max Pairs per Race logic
    # Sort by EV descending within race and take top K
    # Using groupby head
    sel = sel.sort_values(['race_id', 'ev'], ascending=[True, False])
    sel = sel.groupby('race_id').head(params['max_pairs'])
    
    # Calc Metrics
    total_bets = len(sel)
    total_return = sel['return'].sum()
    roi = total_return / (total_bets * 100)
    
    # Race Hit Rate
    # Race is HIT if sum(return) > 0 for that race? Or strictly hit?
    # Umaren is 1 winner. So if return > 0, it's a hit.
    race_groups = sel.groupby('race_id')['return'].sum()
    races_bet = len(race_groups)
    races_hit = (race_groups > 0).sum()
    race_hit_rate = races_hit / races_bet if races_bet > 0 else 0
    
    # Hit Day Rate
    sel['date_str'] = sel['date'].astype(str) # ensure string
    day_groups = sel.groupby('date_str')['return'].sum()
    days_bet = len(day_groups)
    days_hit = (day_groups > 0).sum()
    hit_day_rate = days_hit / days_bet if days_bet > 0 else 0
    
    # Avg Days Per Hit (approx)
    # Total Days / Days Hit? No, Days Bet / Days Hit * (Avg interval).
    # Just Days Bet / Days Hit (frequency in bet days).
    # If we want calendar days, it's harder.
    # User asks "Avg days per hit" (Interval). 
    # Approx: 1 / Hit Day Rate (if betting every day).
    # For JRA (weekend), 1 / Rate * (Days between races). 
    # Let's just report Hit Day Rate.
    
    # Min ROI Year
    years = sel['year'].unique()
    yearly_rois = []
    for y in years:
        ydf = sel[sel['year'] == y]
        if len(ydf) > 0:
            yroi = ydf['return'].sum() / (len(ydf) * 100)
            yearly_rois.append(yroi)
        else:
            yearly_rois.append(0.0)
    min_year_roi = min(yearly_rois) if yearly_rois else 0.0

    return {
        'e_th': params['e_th'],
        'p_th': params['p_th'],
        'max_pairs': params['max_pairs'],
        'bets': total_bets,
        'roi': roi,
        'min_year_roi': min_year_roi,
        'race_hit_rate': race_hit_rate,
        'hit_day_rate': hit_day_rate,
        'races_bet': races_bet
    }

def optimize(input_path: str):
    logger.info("Starting Optimization...")
    df = pd.read_parquet(input_path)
    
    # Train: 2022-2023
    train_df = df[df['year'].isin([2022, 2023])].copy()
    
    # Grid Search
    # EV: [4, 5, 6, 8, 10]
    # P: [0.3%, 0.5%, 0.8%, 1.0%, 1.5%, 2.0%] -> [0.003, 0.005, 0.008, 0.010, 0.015, 0.020]
    # MaxPairs: [1, 2, 3]
    
    e_ths = [4.0, 5.0, 6.0, 8.0, 10.0]
    p_ths = [0.003, 0.005, 0.008, 0.010, 0.015, 0.020]
    max_pairs_list = [1, 2, 3]
    
    results = []
    
    total_combs = len(e_ths) * len(p_ths) * len(max_pairs_list)
    logger.info(f"Evaluating {total_combs} combinations on Train Data...")
    
    for e, p, k in itertools.product(e_ths, p_ths, max_pairs_list):
        params = {'e_th': e, 'p_th': p, 'max_pairs': k}
        metrics = evaluate_strategy(train_df, params)
        if metrics['bets'] > 0:
            results.append(metrics)
            
    res_df = pd.DataFrame(results)
    res_df.to_csv("reports/simulations/umaren_optimization_train.csv", index=False)
    
    # Find Pareto Candidates (ROI vs HitDayRate)
    # Filter minimum requirements
    # RaceHitRate >= 3% (0.03)
    # HitDayRate >= 15% (0.15)
    # ROI >= 100% (1.0)
    
    valid_configs = res_df[
        (res_df['race_hit_rate'] >= 0.03) &
        (res_df['hit_day_rate'] >= 0.15) &
        (res_df['roi'] >= 1.0)
    ].copy()
    
    logger.info(f"Found {len(valid_configs)} valid configurations in Train.")
    
    # Verify on Test (2024)
    test_df = df[df['year'] == 2024].copy()
    test_results = []
    
    for _, row in valid_configs.iterrows():
        params = {
            'e_th': row['e_th'],
            'p_th': row['p_th'],
            'max_pairs': int(row['max_pairs'])
        }
        m = evaluate_strategy(test_df, params)
        m['train_roi'] = row['roi']
        m['train_hit_day'] = row['hit_day_rate']
        test_results.append(m)
        
    test_res_df = pd.DataFrame(test_results)
    test_res_df.to_csv("reports/simulations/umaren_optimization_test_2024.csv", index=False)
    logger.info("Optimization Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", action="store_true")
    parser.add_argument("--opt", action="store_true")
    parser.add_argument("--input", default="reports/simulations/v13_e1_enriched_2022_2024.parquet")
    parser.add_argument("--candidates", default="reports/simulations/umaren_candidates.parquet")
    args = parser.parse_args()
    
    if args.gen:
        generate_candidates(args.input, args.candidates)
    
    if args.opt:
        optimize(args.candidates)

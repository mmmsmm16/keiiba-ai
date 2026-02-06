import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd

from src.utils.payout_loader import PayoutLoader, format_combination

import numpy as np
import logging
import itertools
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_umaren_odds(odds_str: str) -> Dict[Tuple[int, int], float]:
    """
    Parse Chunk-13 Umaren odds string: [Pair 4][Odds 6][Pop 3]
    Returns: {(horse_i, horse_j): odds}
    """
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
            # pop = int(block[10:13]) # unused
            
            # Key is sorted tuple
            key = tuple(sorted((h1, h2)))
            res[key] = odds_val
        except:
            continue
            
    return res

def calculate_harville_probs(p_cal: np.array, horse_numbers: np.array) -> Dict[Tuple[int, int], float]:
    """
    Calculate Umaren probabilities using Harville approximation.
    P({i, j}) = P(i, j) + P(j, i)
    P(i, j) = p_i * p_j / (1 - p_i)
    """
    res = {}
    n = len(p_cal)
    if n < 2: return {}
    
    # Normalize p_cal to sum to 1.0 (just in case, though calibration should be close)
    # But usually Harville assumes p sums to 1.
    p_sum = np.sum(p_cal)
    if p_sum <= 0: return {}
    p_norm = p_cal # / p_sum # Use raw calibrated probs? 
    # If p_cal sum < 1 (due to track takeout removed? No, p_cal is probability), 
    # usually it sums to ~1.0. If not, Harville might be skewed.
    # Let's assume p_cal is proper probability.
    
    for i in range(n):
        for j in range(n):
            if i == j: continue
            
            pi = p_norm[i]
            pj = p_norm[j]
            den = 1.0 - pi
            if den <= 0: den = 1e-9
            
            p_ij = pi * (pj / den)
            
            h1 = horse_numbers[i]
            h2 = horse_numbers[j]
            key = tuple(sorted((h1, h2)))
            
            res[key] = res.get(key, 0.0) + p_ij
            
    return res

def run_multiticket_optimization(input_path: str):
    logger.info(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path)
    
    # Load Payout Map (for validation)
    years = [2022, 2023, 2024]
    logger.info(f"Loading Payout Map for {years}...")
    loader = PayoutLoader()
    payout_map = loader.load_payout_map(years)
    
    # Filter valid years (Train: 2022-2023)
    train_df = df[df['year_valid'].isin([2022, 2023])].copy()
    
    logger.info(f"Processing {len(train_df)} rows for optimization...")
    
    # Pre-calculate unique race IDs
    race_ids = train_df['race_id'].unique()
    
    results = []
    
    # Define Strategies parameters
    # Mode B (Odds-Aware) Check
    ev_thresholds_win = [2.0, 2.5, 3.0]
    ev_thresholds_place = [1.05, 1.1, 1.2]
    ev_thresholds_umaren = [15.0, 20.0, 30.0] # High payout target
    
    # Simulation Loop
    # Due to complexity, we run one pass per race and collect "potential bets" with their EV and Probs.
    # Then we can filter effectively.
    
    bet_candidates = [] # List of dicts: {race_id, type, target, odds, prob, ev, return, ...}
    
    count = 0
    for rid in race_ids:
        count += 1
        if count % 1000 == 0: logger.info(f"Processed {count}/{len(race_ids)} races...")
        
        race_rows = train_df[train_df['race_id'] == rid]
        if race_rows.empty: continue
        
        # Base Info
        horses = race_rows['horse_number'].values
        p_cals = race_rows['p_cal'].values
        p_wins = race_rows['p_win'].values
        p_places = race_rows['p_place'].values
        year = race_rows['year_valid'].iloc[0]
        
        odds_win = race_rows['odds_win_pre'].values
        
        # Place Odds (Min/Max mean)
        odds_place_min = race_rows['odds_place_pre_min'].values
        odds_place_max = race_rows['odds_place_pre_max'].values
        # Simple average for EV estimation
        odds_place = (np.nan_to_num(odds_place_min) + np.nan_to_num(odds_place_max)) / 2.0
        
        # Umaren Odds Parsing
        umaren_str = race_rows.iloc[0].get('odds_umaren_pre', '')
        umaren_odds_map = parse_umaren_odds(umaren_str)
        
        # Umaren Probs (Use p_win for Harville)
        umaren_probs_map = calculate_harville_probs(p_wins, horses)
        
        # 1. Win Candidates
        for i, h in enumerate(horses):
            p = p_wins[i]
            o = odds_win[i]
            if pd.isna(o) or o == 0: continue
            
            ev = p * o
            
            hit_amount = payout_map.get(rid, {}).get('tansho', {}).get(f"{h:02}", 0)
            
            bet_candidates.append({
                'race_id': rid,
                'type': 'win',
                'target': f"{h:02}", # Force string
                'prob': p,
                'odds': o,
                'ev': ev,
                'return': hit_amount,
                'row_idx': i,
                'year': year
            })

        # 2. Place Candidates
        for i, h in enumerate(horses):
            p = p_places[i]
            o = odds_place[i]
            if pd.isna(o) or o == 0: continue
            
            ev = p * o
            
            # Place hit amount is variable, depends on field size and winners.
            # We use actual payout if hit.
            hit_amount = payout_map.get(rid, {}).get('fukusho', {}).get(f"{h:02}", 0)
            
            bet_candidates.append({
                'race_id': rid,
                'type': 'place',
                'target': f"{h:02}", # Force string
                'prob': p,
                'odds': o,
                'ev': ev,
                'return': hit_amount,
                'row_idx': i,
                'year': year
            })

        # 3. Umaren Candidates
        for pair, p_u in umaren_probs_map.items():
            o_u = umaren_odds_map.get(pair, 0.0)
            if o_u <= 0: continue
            
            ev_u = p_u * o_u
            
            # Filter low potential bets to save space
            # Keep if EV is decent OR Probability is high (for Odds-Free mode)
            if ev_u < 0.7 and p_u < 0.02: continue
            
            # Check Hit
            comb_str = format_combination(list(pair), ordered=False)
            hit_amount = payout_map.get(rid, {}).get('umaren', {}).get(comb_str, 0)
            
            bet_candidates.append({
                'race_id': rid,
                'type': 'umaren',
                'target': comb_str, # "0102"
                'prob': p_u,
                'odds': o_u,
                'ev': ev_u,
                'return': hit_amount,
                'row_idx': 0, # Dummy
                'year': year
            })

    # Save candidates for analysis
    logger.info(f"Generated {len(bet_candidates)} bet candidates.")
    cand_df = pd.DataFrame(bet_candidates)
    output_temp = "reports/simulations/bet_candidates_temp.parquet"
    cand_df.to_parquet(output_temp)
    logger.info(f"Saved candidates to {output_temp}")
    
    return cand_df

def analyze_candidates(df: pd.DataFrame):
    """
    Analyze candidates to find optimal strategies per ticket type.
    """
    logger.info("Analyzing candidates...")
    
    stats = []
    
    def calc_stats(sub_df, label, thresh):
        if sub_df.empty: return
        
        total_bets = len(sub_df)
        total_return = sub_df['return'].sum()
        roi = total_return / (total_bets * 100)
        hit_rate = (sub_df['return'] > 0).mean()
        
        # Yearly ROI
        years = sub_df['year'].unique()
        yearly_rois = []
        for y in years:
            ydf = sub_df[sub_df['year'] == y]
            if len(ydf) > 0:
                yroi = ydf['return'].sum() / (len(ydf) * 100)
                yearly_rois.append(yroi)
            else:
                yearly_rois.append(0.0)
        
        min_roi_yr = min(yearly_rois) if yearly_rois else 0.0
        
        stats.append({
            'label': label,
            'threshold': thresh,
            'bets': total_bets,
            'roi': roi,
            'hit_rate': hit_rate,
            'min_roi_yr': min_roi_yr
        })
        logger.info(f"[{label}] Thresh={thresh}: Bets={total_bets}, ROI={roi*100:.2f}%, MinYr={min_roi_yr*100:.2f}%")

    # 1. Win Strategy
    win_df = df[df['type'] == 'win'].copy()
    if not win_df.empty:
        for ev_min in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
            sel = win_df[win_df['ev'] >= ev_min]
            calc_stats(sel, 'Win_EV', ev_min)
            
    # 2. Place Strategy
    place_df = df[df['type'] == 'place'].copy()
    if not place_df.empty:
        for ev_min in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]:
            sel = place_df[place_df['ev'] >= ev_min]
            calc_stats(sel, 'Place_EV', ev_min)

    # 3. Umaren Strategy
    umaren_df = df[df['type'] == 'umaren'].copy()
    if not umaren_df.empty:
        for ev_min in [1.5, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]:
            sel = umaren_df[umaren_df['ev'] >= ev_min]
            calc_stats(sel, 'Umaren_EV', ev_min)
            
    # Save Stats
    if stats:
        sdf = pd.DataFrame(stats)
        sdf.to_csv("reports/simulations/multiticket_pareto.csv", index=False)
        logger.info("Saved pareto stats to reports/simulations/multiticket_pareto.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", action="store_true", help="Generate candidates")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing candidates")
    parser.add_argument("--input", default="reports/simulations/v13_e1_enriched_2022_2024.parquet")
    args = parser.parse_args()
    
    if args.gen:
        df = run_multiticket_optimization(args.input)
        analyze_candidates(df)
    elif args.analyze:
        df = pd.read_parquet("reports/simulations/bet_candidates_temp.parquet")
        analyze_candidates(df)
    else:
        # Default run all
        try:
             df = pd.read_parquet("reports/simulations/bet_candidates_temp.parquet")
             analyze_candidates(df)
        except:
             df = run_multiticket_optimization(args.input)
             analyze_candidates(df)


import sys
import os
import logging
import pandas as pd
import numpy as np
from itertools import combinations, permutations
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from utils.payout_loader import PayoutLoader, check_hit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Starting Multi-Bet ROI Simulation (2024)...")
    
    # 1. Load Predictions (Base Model)
    pred_path = "data/temp_t2/T2_predictions_2024_2025.parquet"
    if not os.path.exists(pred_path):
        logger.error(f"Predictions not found: {pred_path}")
        return
    df_pred = pd.read_parquet(pred_path)
    df_pred['race_id'] = df_pred['race_id'].astype(str)
    
    # Filter for 2024
    df_pred['date'] = pd.to_datetime(df_pred['date'])
    df_2024 = df_pred[df_pred['date'].dt.year == 2024].copy()
    logger.info(f"Analyzing 2024 data: {len(df_2024)} rows")
    
    # 2. Load Payout Map
    loader = PayoutLoader()
    payout_map = loader.load_payout_map([2024])
    logger.info(f"Loaded payout map for {len(payout_map)} races")
    
    # 3. Strategy Definitions
    # (Label, TicketType, Method, threshold, box_n/axis_th)
    strategies = []
    
    # Thresholds to test
    confidence_levels = [0.3, 0.4, 0.5]
    
    # --- UmaRen (Exacta Box) ---
    # Box 2-5 horses if their score >= th
    for th in confidence_levels:
        for n in [2, 3, 4, 5]:
            strategies.append({
                'name': f"UmaRen Box{n} (Conf>={th})",
                'type': 'umaren',
                'method': 'box',
                'n': n,
                'th': th
            })
            
    # --- Wide (Wide Box) ---
    for th in confidence_levels:
        for n in [2, 3, 4, 5]:
            strategies.append({
                'name': f"Wide Box{n} (Conf>={th})",
                'type': 'wide',
                'method': 'box',
                'n': n,
                'th': th
            })

    # --- Sanrenpuku (Trio Box) ---
    for th in [0.3, 0.4]: # Trio harder to hit with high conf
        for n in [3, 4, 5]:
            strategies.append({
                'name': f"Trio Box{n} (Conf>={th})",
                'type': 'sanrenpuku',
                'method': 'box',
                'n': n,
                'th': th
            })
            
    # --- Sanrentan (Trifecta Nagashi) ---
    # Axis: Top 1 (Must be > AxisTh), Flow: Next N (Must be > FlowTh)
    for axis_th in [0.5, 0.6]:
        for n_flow in [3, 4, 5]:
            strategies.append({
                'name': f"TriFecta 1-Axis->{n_flow} (Ax>={axis_th})",
                'type': 'sanrentan',
                'method': 'nagashi_1',
                'n': n_flow,
                'th': 0.1, # Flow just needs to be top N
                'axis_th': axis_th
            })
    
    results = []
    
    # Group by race
    grouped = df_2024.groupby('race_id')
    
    # Run simulation
    logger.info(f"Simulating {len(strategies)} strategies...")
    
    # Initialize aggregators
    # key: strategy_index, value: {'stake': 0, 'return': 0, 'hits': 0, 'bets': 0}
    agg = {i: {'stake': 0, 'return': 0, 'hits': 0, 'bets': 0} for i in range(len(strategies))}
    
    for race_id, group in tqdm(grouped):
        if race_id not in payout_map:
            continue
            
        # Sort by prediction score descending
        sorted_group = group.sort_values('pred_prob', ascending=False)
        top_horses = sorted_group['horse_number'].tolist()
        scores = sorted_group['pred_prob'].tolist()
        
        for i, strat in enumerate(strategies):
            ticket_type = strat['type']
            method = strat['method']
            th = strat['th']
            n = strat['n']
            
            bet_combinations = []
            
            if method == 'box':
                # Select horses that meet threshold and within top N
                valid_indices = [k for k in range(len(scores)) if scores[k] >= th]
                selected_indices = valid_indices[:n] # Top N among filtered
                
                # If we don't have enough horses for the box, skip or buy fewer?
                # Usually Box N implies we buy exactly N if available. 
                # Strict Rule: Must have exactly N horses? Or Up to N?
                # Interpretation: "Buy Top N, provided they are above TH".
                # If filtered count < required for ticket (e.g. 2 for UmaRen), no bet.
                
                min_req = 2 if ticket_type in ['umaren', 'wide'] else 3
                if len(selected_indices) < min_req:
                    continue
                
                selected_horses = [top_horses[k] for k in selected_indices]
                
                # Generate combinations
                if ticket_type in ['umaren', 'wide']:
                    combos = list(combinations(selected_horses, 2))
                elif ticket_type == 'sanrenpuku':
                    combos = list(combinations(selected_horses, 3))
                
                for c in combos:
                    bet_combinations.append(c)

            elif method == 'nagashi_1':
                # Trifecta Nagashi: 1st (Axis) -> 2nd/3rd (Flow)
                axis_th = strat['axis_th']
                
                # Check Axis
                if scores[0] < axis_th:
                    continue
                
                axis_horse = top_horses[0]
                
                # Flow horses: Next N
                flow_indices = list(range(1, 1+n))
                # Ensure we have enough horses
                if len(top_horses) < 1+n:
                    continue
                    
                flow_horses = [top_horses[k] for k in flow_indices]
                
                # Permutations: Axis -> Flow1 -> Flow2
                # Sanrentan: 1st=Axis, 2nd=FlowA, 3rd=FlowB
                # Flow permutations (select 2 from flow list)
                flow_perms = list(permutations(flow_horses, 2))
                
                for p in flow_perms:
                    # (Axis, Flow1, Flow2)
                    bet_combinations.append((axis_horse, p[0], p[1]))
            
            # Purchase
            if not bet_combinations:
                continue
                
            cost = len(bet_combinations) * 100
            payout = 0
            
            ordered = (ticket_type == 'sanrentan')
            
            for combo in bet_combinations:
                pay = check_hit(list(combo), payout_map, race_id, ticket_type, ordered=ordered)
                payout += pay
            
            agg[i]['stake'] += cost
            agg[i]['return'] += payout
            agg[i]['bets'] += 1
            if payout > 0:
                agg[i]['hits'] += 1

    # Compile Results
    final_results = []
    for i, strat in enumerate(strategies):
        s = agg[i]
        stake = s['stake']
        ret = s['return']
        roi = (ret / stake * 100) if stake > 0 else 0
        hit_rate = (s['hits'] / s['bets'] * 100) if s['bets'] > 0 else 0
        
        final_results.append({
            'Strategy': strat['name'],
            'Ticket': strat['type'],
            'ROI': roi,
            'Profit': ret - stake,
            'Bets': s['bets'],
            'HitRate': hit_rate,
            'Stake': stake
        })
        
    df_res = pd.DataFrame(final_results)
    df_res = df_res.sort_values('ROI', ascending=False)
    
    print("\n========= Multi-Bet ROI Simulation Results (Top 20) =========")
    print(df_res.head(20).to_string(index=False))
    
    # Save
    os.makedirs("data/reports", exist_ok=True)
    df_res.to_csv("data/reports/multi_bet_roi_2024.csv", index=False)
    logger.info("Saved report to data/reports/multi_bet_roi_2024.csv")

if __name__ == "__main__":
    main()

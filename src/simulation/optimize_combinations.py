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

def run_simulation(input_path: str, output_path_base: str):
    logger.info(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path)
    
    # Filter valid years
    years = [2022, 2023, 2024]
    df = df[df['year_valid'].isin(years)].copy()
    
    # Load Payouts
    logger.info("Loading Payout Map...")
    loader = PayoutLoader()
    payout_map = loader.load_payout_map(years)
    
    race_ids = df['race_id'].unique()
    
    # Simulation Results Holder
    # structure: {rule_name: {year: {bets, return, hit_races, hit_days, total_races}}}
    results = {}
    
    # Pre-define Rules
    # WIDE
    wide_rules = [
        {'id': 'W1', 'name': 'Wide_1-2', 'type': 'wide', 'buy': lambda r: [(r[0], r[1])]},
        {'id': 'W2', 'name': 'Wide_Top3_Box', 'type': 'wide', 'buy': lambda r: list(itertools.combinations(r[:3], 2))},
        {'id': 'W4', 'name': 'Wide_1-Top4', 'type': 'wide', 'buy': lambda r: [(r[0], x) for x in r[1:4]]},
        {'id': 'W5', 'name': 'Wide_2-Top5', 'type': 'wide', 'buy': lambda r: [(r[0], x) for x in r[1:5]] + [(r[1], x) for x in r[2:5]]}, # Actually user said 2 Axis? "2軸(Top2固定)-相手Top5"? ambiguous.
        # User spec: "2軸（Top2固定）- 相手Top5（3点：1-3,2-3,1-4 等のパターンも比較）"
        # Let's verify spec: "Top2固定" usually means 1-X and 2-X. 
        # Interpretation: Pair (1, X) and (2, X) where X in 3..5?
        # Let's stick to standard flow: "1軸相手N頭" or "Box".
        # W5 (Alternative): 1-Top5 (4 bets)
        {'id': 'W_1_5', 'name': 'Wide_1-Top5', 'type': 'wide', 'buy': lambda r: [(r[0], x) for x in r[1:5]]},
    ]
    
    # TRIO (Sanrenpuku)
    trio_rules = [
        {'id': 'T3F1', 'name': 'Trio_1-2-3', 'type': 'sanrenpuku', 'buy': lambda r: [tuple(sorted(r[:3]))]},
        {'id': 'T3F2', 'name': 'Trio_1-Top4_Ax1', 'type': 'sanrenpuku', 'buy': lambda r: [tuple(sorted((r[0], x, y))) for x, y in itertools.combinations(r[1:4], 2)]},
        {'id': 'T3F5', 'name': 'Trio_Top5_Box', 'type': 'sanrenpuku', 'buy': lambda r: list(itertools.combinations(r[:5], 3))},
    ]
    
    # TRIFECTA (Sanrentan)
    # T3T1: 1->2->3
    # T3T2: 1 -> 2,3 -> 2,3,4 (Formation?) 
    # User spec: "1着固定（Top1）2-3着はTop3/4で組む"
    # Let's do 1 -> Top3 -> Top3 (minus 1)
    trifecta_rules = [
        {'id': 'T3T1', 'name': 'Tri_1-2-3', 'type': 'sanrentan', 'buy': lambda r: [(r[0], r[1], r[2])]},
        {'id': 'T3T2', 'name': 'Tri_1>234>234', 'type': 'sanrentan', 'buy': lambda r: [(r[0], x, y) for x in r[1:4] for y in r[1:4] if x != y]},
    ]

    all_strategies = wide_rules + trio_rules + trifecta_rules
    
    # define Filters
    # p1 >= t1, margin >= t2, entropy <= t3
    # Grid Search
    p1_ths = [0.2, 0.3, 0.4]
    margin_ths = [0.0, 0.05, 0.1]
    entropy_ths = [999.0, 1.5] # 999 is no filter
    
    # To save memory, we accumulate stats per (Strategy, FilterCombo, Year)
    # FilterCombo Key: "p0.3_m0.05_e1.5"
    
    logger.info("Iterating races...")
    
    # Pre-calculated stats accumulator
    # keys: (strat_id, filter_key, year)
    stats_acc = {} 
    
    debug_limit = 0 
    
    count = 0
    for rid in race_ids:
        count += 1
        if count % 2000 == 0: logger.info(f"Processed {count}/{len(race_ids)}")
        
        race_df = df[df['race_id'] == rid]
        if race_df.empty: continue
        
        # Extract features
        row1 = race_df.iloc[0]
        p1 = row1.get('p1', 0)
        margin = row1.get('margin', 0)
        entropy = row1.get('entropy', 999)
        year = row1.get('year_valid', 2000)
        date_str = str(row1.get('date', ''))
        
        # Sort horses by p_cal desc
        race_df_sorted = race_df.sort_values('p_cal', ascending=False)
        top_horses = race_df_sorted['horse_number'].values # 1-based integers
        
        if len(top_horses) < 5: continue # Skip small fields for sim
        
        # Prepare Combinations per Rule type
        # Wide: (A, B) sorted
        # Trio: (A, B, C) sorted
        # Trifecta: (A, B, C) ordered
        
        for p_th in p1_ths:
            if p1 < p_th: continue
            for m_th in margin_ths:
                if margin < m_th: continue
                for e_th in entropy_ths:
                    if entropy > e_th: continue
                    
                    filter_key = f"p{p_th:.2f}_m{m_th:.2f}_e{e_th:.1f}"
                    
                    for rule in all_strategies:
                        # Skip expensive rules if p1 is low (heuristic)
                        if rule['type'] == 'sanrentan' and p1 < 0.3: continue
                        
                        # Generate Bet List
                        bets = rule['buy'](top_horses)
                        bet_count = len(bets)
                        bet_amt = bet_count * 100 # 100 yen per bet
                        
                        # Calculate Return
                        total_ret = 0
                        hit = 0
                        
                        pm = payout_map.get(rid, {})
                        
                        type_key = rule['type']
                        # map to payout_loader keys if needed
                        # payout_loader uses: 'wide', 'sanrenpuku', 'sanrentan'
                        # wide keys: "0102"
                        # trio keys: "010203"
                        # tri keys: "010203"
                        
                        pay_dict = pm.get(type_key, {})
                        
                        for b in bets:
                            if type_key in ['wide', 'sanrenpuku']:
                                # Ordered=False
                                k = format_combination(list(b), ordered=False)
                            else:
                                # Ordered=True
                                k = format_combination(list(b), ordered=True)
                            
                            pay = pay_dict.get(k, 0)
                            if pay > 0:
                                total_ret += pay
                                hit = 1
                        
                        # Update Stats
                        # (strat_id, filter_key, year)
                        k = (rule['id'], filter_key, year)
                        if k not in stats_acc:
                            stats_acc[k] = {
                                'bets': 0, 
                                'return': 0, 
                                'races': 0, 
                                'hit_races': 0,
                                'dates': set(),
                                'hit_dates': set()
                            }
                        
                        s = stats_acc[k]
                        s['bets'] += bet_count
                        s['return'] += total_ret
                        s['races'] += 1
                        if hit > 0:
                            s['hit_races'] += 1
                            s['hit_dates'].add(date_str)
                        s['dates'].add(date_str)

    # Convert to DataFrame
    logger.info("Aggregating results...")
    rows = []
    
    for (sid, fkey, yr), s in stats_acc.items():
        hit_day_rate = len(s['hit_dates']) / len(s['dates']) if len(s['dates']) > 0 else 0
        roi = s['return'] / (s['bets'] * 100) if s['bets'] > 0 else 0
        race_hit_rate = s['hit_races'] / s['races'] if s['races'] > 0 else 0
        
        rows.append({
            'strategy': sid,
            'filter': fkey,
            'year': yr,
            'bets': s['bets'],
            'return': s['return'],
            'roi': roi,
            'race_hit_rate': race_hit_rate,
            'hit_day_rate': hit_day_rate,
            'races_bet': s['races'],
            'days_bet': len(s['dates'])
        })
        
    res_df = pd.DataFrame(rows)
    # Pivot or Aggregate to see Train/Test
    # We want rows: Strategy+Filter
    # Cols: ROI_2022, ROI_2023, ROI_2024, etc.
    
    res_df.to_csv(output_path_base + "_raw.csv", index=False)
    
    # Summary
    # Group by Strategy+Filter
    summary = []
    groups = res_df.groupby(['strategy', 'filter'])
    
    for (strat, filt), g in groups:
        # Train (2022+2023)
        g_train = g[g['year'].isin([2022, 2023])]
        bets_tr = g_train['bets'].sum()
        ret_tr = g_train['return'].sum()
        roi_tr = ret_tr / (bets_tr * 100) if bets_tr > 0 else 0
        
        # Test (2024)
        g_test = g[g['year'] == 2024]
        bets_te = g_test['bets'].sum()
        ret_te = g_test['return'].sum()
        roi_te = ret_te / (bets_te * 100) if bets_te > 0 else 0
        
        # Min Yearly ROI
        min_yr = g['roi'].min()
        
        # Avg Hit Day Rate (Weighted?) Just take Test
        hit_day_te = g_test['hit_day_rate'].mean() if not g_test.empty else 0
        
        summary.append({
            'strategy': strat,
            'filter': filt,
            'roi_train': roi_tr,
            'roi_test': roi_te,
            'bets_test': bets_te,
            'hit_day_test': hit_day_te,
            'min_year_roi': min_yr
        })
    
    sum_df = pd.DataFrame(summary)
    sum_df.to_csv(output_path_base + "_summary.csv", index=False)
    logger.info(f"Saved summary to {output_path_base}_summary.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/simulations/v13_e1_enriched_2022_2024.parquet")
    parser.add_argument("--out", default="reports/simulations/i2_combinations")
    args = parser.parse_args()
    
    run_simulation(args.input, args.out)

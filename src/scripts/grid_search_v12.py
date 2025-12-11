"""
Grid Search & Backtest for v12 Strategy (Segmented Analysis)
- Define Race Segments (e.g., High EV, Favorite, Longshot)
- For each segment, find the Best Strategy (Bet Type + Opponent Count)
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
from tabulate import tabulate
from itertools import product
from tqdm import tqdm

# パス設定
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, project_root)

from src.pipeline.evaluate import load_payout_data

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_predictions(exp_name):
    pred_path = os.path.join(project_root, 'experiments', exp_name, 'reports', 'predictions.parquet')
    if not os.path.exists(pred_path):
        logger.error(f"Predictions file not found: {pred_path}")
        return None
    
    df = pd.read_parquet(pred_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['date', 'race_id'])
    
    df['pred_rank'] = df.groupby('race_id')['score'].rank(method='first', ascending=False)
    
    if 'expected_value' not in df.columns and 'odds' in df.columns:
         df['expected_value'] = df['score'] * df['odds']
         
    if 'rank' not in df.columns and 'target' in df.columns:
        df['rank'] = df['target'].map({3: 1, 2: 2, 1: 3}).fillna(99)
            
    return df

def build_payout_map(payout_df):
    p_map = {}
    for _, row in payout_df.iterrows():
        rid = row['race_id']
        p_map[rid] = {
            'tansho': {}, 'umaren': {}, 'wide': {}, 
            'sanrenpuku': {}, 'sanrentan': {}
        }
        # Tansho
        for i in range(1, 4):
            k, v = f'haraimodoshi_tansho_{i}a', f'haraimodoshi_tansho_{i}b'
            if row.get(k):
                try: p_map[rid]['tansho'][str(int(row[k])).zfill(2)] = int(row[v])
                except: pass
        # Umaren
        for i in range(1, 4):
            k, v = f'haraimodoshi_umaren_{i}a', f'haraimodoshi_umaren_{i}b'
            if row.get(k):
                try: p_map[rid]['umaren'][str(row[k]).zfill(4)] = int(row[v])
                except: pass
        # Wide
        for i in range(1, 8):
            k, v = f'haraimodoshi_wide_{i}a', f'haraimodoshi_wide_{i}b'
            if row.get(k):
                try: p_map[rid]['wide'][str(row[k]).zfill(4)] = int(row[v])
                except: pass
        # Sanrenpuku
        for i in range(1, 4):
            k, v = f'haraimodoshi_sanrenpuku_{i}a', f'haraimodoshi_sanrenpuku_{i}b'
            if row.get(k):
                try: p_map[rid]['sanrenpuku'][str(row[k]).zfill(6)] = int(row[v])
                except: pass
        # Sanrentan
        for i in range(1, 7):
            k, v = f'haraimodoshi_sanrentan_{i}a', f'haraimodoshi_sanrentan_{i}b'
            if row.get(k):
                try: p_map[rid]['sanrentan'][str(row[k]).zfill(6)] = int(row[v])
                except: pass
    return p_map

def simulate_segment(df, p_map, strategy_type, n_opps):
    """
    Simulate a strategy on the GIVEN DATAFRAME (Segment).
    Returns basic stats.
    """
    total_bet = 0
    total_return = 0
    races = 0
    hits = 0
    
    # Pre-calculated groups usually passed, but here we group again for safety
    # or assume df is already filtered. We iterate by races in df.
    
    # Optimization: Filter p_map keys once? No, cheap dict lookup.
    
    race_groups = df.groupby('race_id')
    
    for race_id, group in race_groups:
        if race_id not in p_map: continue
        
        group = group.sort_values('pred_rank')
        if len(group) < 1 + n_opps: continue
        
        axis = group.iloc[0]
        axis_num = int(axis['horse_number'])
        opps = group.iloc[1:1+n_opps]
        opp_nums = [int(x) for x in opps['horse_number']]
        
        payouts = p_map[race_id]
        current_return = 0
        current_bet = 0
        hit_flg = 0
        
        if strategy_type == 'umaren':
            current_bet = len(opp_nums) * 100
            for opp in opp_nums:
                key = "".join(f"{x:02}" for x in sorted([axis_num, opp]))
                if key in payouts['umaren']:
                    current_return += payouts['umaren'][key]
                    hit_flg = 1
                    
        elif strategy_type == 'wide':
            current_bet = len(opp_nums) * 100
            for opp in opp_nums:
                key = "".join(f"{x:02}" for x in sorted([axis_num, opp]))
                if key in payouts['wide']:
                    current_return += payouts['wide'][key]
                    hit_flg = 1
        
        elif strategy_type == 'sanrenpuku':
            # Axis 1 head -> Opps (2 needed)
            from itertools import combinations
            combs = list(combinations(opp_nums, 2))
            current_bet = len(combs) * 100
            for pair in combs:
                key = "".join(f"{x:02}" for x in sorted([axis_num, pair[0], pair[1]]))
                if key in payouts['sanrenpuku']:
                    current_return += payouts['sanrenpuku'][key]
                    hit_flg = 1
        
        elif strategy_type == 'sanrentan':
            # Axis 1st Fixed -> Opps (2 needed)
            from itertools import permutations
            perms = list(permutations(opp_nums, 2))
            current_bet = len(perms) * 100
            for pair in perms:
                key = f"{axis_num:02}{pair[0]:02}{pair[1]:02}"
                if key in payouts['sanrentan']:
                    current_return += payouts['sanrentan'][key]
                    hit_flg = 1
        
        total_bet += current_bet
        total_return += current_return
        races += 1
        if hit_flg > 0: hits += 1
            
    return {
        'roi': (total_return / total_bet * 100) if total_bet > 0 else 0,
        'profit': total_return - total_bet,
        'hit_rate': (hits / races * 100) if races > 0 else 0,
        'bets': total_bet,
        'races': races
    }

def get_segments(df):
    """
    Define mutually exclusive (or overlapping) segments for analysis.
    Here we define overlapping segments to find best conditions.
    """
    # Pre-calculate race-level features
    # Axis (Top1) features
    cols = ['race_id', 'odds', 'expected_value', 'score']
    if 'popularity' in df.columns:
        cols.append('popularity')
        
    top1 = df[df['pred_rank'] == 1][cols].set_index('race_id')
    # Rename for clarity
    rename_map = {'odds': 'axis_odds', 'expected_value': 'axis_ev', 'score': 'axis_score'}
    if 'popularity' in df.columns:
        rename_map['popularity'] = 'axis_pop'
        
    top1 = top1.rename(columns=rename_map)
    
    # Helper to safe get
    def get_val(rid, col, default=0):
        if rid not in top1.index: return default
        return top1.loc[rid].get(col, default)

    return [
        {
            'name': 'High Value (EV >= 1.2)',
            'filter': lambda rid: get_val(rid, 'axis_ev') >= 1.2
        },
        {
            'name': 'Mid Value (0.8 <= EV < 1.2)',
            'filter': lambda rid: 0.8 <= get_val(rid, 'axis_ev') < 1.2
        },
        {
            'name': 'Low Popularity (Pop >= 7)',
            'filter': lambda rid: get_val(rid, 'axis_pop') >= 7
        },
        {
            'name': 'Longshot (Odds >= 20.0)',
            'filter': lambda rid: get_val(rid, 'axis_odds') >= 20.0
        },
        {
            'name': 'Any (Baseline)',
            'filter': lambda rid: True
        }
    ]


def main():
    parser = argparse.ArgumentParser(description="Segment Optimization for v12")
    parser.add_argument("experiment_name", type=str, default="v12_tabnet_revival", nargs="?")
    args = parser.parse_args()
    
    df = load_predictions(args.experiment_name)
    if df is None: return

    # Data Split
    df = df[df['date'].dt.year == 2025].copy()
    train_df_all = df[df['date'] <= '2025-03-31']
    test_df_all = df[df['date'] >= '2025-04-01']
    
    # Load Payouts
    payout_df = load_payout_data([2025])
    p_map = build_payout_map(payout_df)
    
    # Segments
    segments = get_segments(df) # Logic is shared, filters applied per race_id
    
    # Grid Search Space
    bet_types = ['umaren', 'wide', 'sanrenpuku', 'sanrentan']
    
    print(f"\nEvaluating {len(segments)} Segments...")
    
    summary_table = []
    
    for seg in segments:
        seg_name = seg['name']
        print(f"\nProcessing Segment: {seg_name}")
        
        # Filter DataFrames
        # This is slow if done race-by-race. Better: Get valid race_ids first.
        # But for clarity, we use the filter function.
        # Optim: Apply filter to unique race_ids
        
        # Train Filter
        train_rids = train_df_all['race_id'].unique()
        train_valid_rids = [rid for rid in train_rids if seg['filter'](rid)]
        seg_train_df = train_df_all[train_df_all['race_id'].isin(set(train_valid_rids))]
        
        # Test Filter
        test_rids = test_df_all['race_id'].unique()
        test_valid_rids = [rid for rid in test_rids if seg['filter'](rid)]
        seg_test_df = test_df_all[test_df_all['race_id'].isin(set(test_valid_rids))]
        
        if len(seg_train_df) == 0:
            print("  No train data.")
            continue
            
        # Grid Search on Train - Find best PER TYPE
        best_configs = {} # type -> best config
        
        for b_type in bet_types:
            best_roi = -1
            best_cfg = None
            
            max_n = 10 if b_type in ['umaren', 'wide'] else 8
            for n in range(1, max_n):
                res = simulate_segment(seg_train_df, p_map, b_type, n)
                
                # Min races filter
                if res['races'] < 30: continue
                
                if res['roi'] > best_roi:
                    best_roi = res['roi']
                    best_cfg = {'type': b_type, 'n': n, 'train_stats': res}
            
            if best_cfg:
                best_configs[b_type] = best_cfg
        
        # Validation & Report
        for b_type in bet_types:
            if b_type in best_configs:
                c = best_configs[b_type]
                test_res = simulate_segment(seg_test_df, p_map, c['type'], c['n'])
                
                # Report only reasonable ROI > 80% to keep table clean? 
                # No, user wants to see umaren specifically.
                
                print(f"  {b_type.upper()}: Top1->{c['n']} | Train: {c['train_stats']['roi']:.1f}% | Test: {test_res['roi']:.1f}%")
                
                summary_table.append([
                    seg_name,
                    f"{b_type} (Opp:{c['n']})",
                    f"{c['train_stats']['roi']:.1f}%",
                    f"{c['train_stats']['hit_rate']:.1f}%",
                    f"{c['train_stats']['races']}",
                    f"{test_res['roi']:.1f}%",
                    f"{test_res['hit_rate']:.1f}%",
                    f"{int(test_res['profit']):,}",
                    f"{test_res['races']}"
                ])
                
    print("\n=== Optimal Strategy by Condition & Type (v12) ===")
    print(tabulate(summary_table, headers=['Condition', 'Strategy', 'Train ROI', 'Hit1Q', 'Races1Q', 'Test ROI', 'Hit2Q+', 'Profit2Q+', 'Races2Q+'], tablefmt='github'))

if __name__ == "__main__":
    main()

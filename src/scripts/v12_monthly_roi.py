"""
v12 EV Strategy - 2025 Monthly ROI Report
Calculates investment, return, profit, and ROI per month using the v12 EV-based strategy.
"""

import pandas as pd
import numpy as np
import os
from itertools import permutations, combinations
from scipy.special import softmax

# Paths
PROJECT_ROOT = '/workspace'
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, 'experiments')

def load_data():
    """Load predictions and payout data."""
    # Load v12 predictions
    pred_path = os.path.join(EXPERIMENTS_DIR, 'v12_tabnet_revival', 'reports', 'predictions.parquet')
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Predictions not found: {pred_path}")
    
    df = pd.read_parquet(pred_path)
    df['date'] = pd.to_datetime(df['date'])
    df['pred_rank'] = df.groupby('race_id')['score'].rank(method='first', ascending=False)
    
    # Load payout data
    payout_paths = [
        os.path.join(EXPERIMENTS_DIR, 'payouts_2024_2025.parquet'),
        os.path.join(EXPERIMENTS_DIR, 'payouts_2024.parquet'),
        os.path.join(EXPERIMENTS_DIR, 'v7_ensemble_full', 'data', 'payout_data.parquet'),
    ]
    
    payout_df = None
    for path in payout_paths:
        if os.path.exists(path):
            payout_df = pd.read_parquet(path)
            print(f"Loaded payout data from: {path}")
            break
    
    if payout_df is None:
        raise FileNotFoundError("Payout data not found")
    
    return df, payout_df

def build_payout_map(payout_df):
    """Build payout lookup map."""
    payout_map = {}
    for _, row in payout_df.iterrows():
        rid = str(row.get('race_id', ''))
        if not rid:
            continue
        if rid not in payout_map:
            payout_map[rid] = {'tansho': {}, 'sanrentan': {}, 'sanrenpuku': {}}
        
        # Sanrentan
        for k in range(1, 7):
            comb = row.get(f'haraimodoshi_sanrentan_{k}a')
            pay = row.get(f'haraimodoshi_sanrentan_{k}b')
            if comb and pay:
                try:
                    payout_map[rid]['sanrentan'][str(comb).strip()] = int(float(pay))
                except:
                    pass
        
        # Sanrenpuku
        for k in range(1, 4):
            comb = row.get(f'haraimodoshi_sanrenpuku_{k}a')
            pay = row.get(f'haraimodoshi_sanrenpuku_{k}b')
            if comb and pay:
                try:
                    payout_map[rid]['sanrenpuku'][str(comb).strip()] = int(float(pay))
                except:
                    pass
    
    return payout_map

def calculate_ev_strategy(df, payout_map, year=2025):
    """Calculate monthly ROI using v12 EV strategy."""
    # Filter to specified year
    df_year = df[df['date'].dt.year == year].copy()
    
    monthly_stats = {}
    
    for race_id, group in df_year.groupby('race_id'):
        race_id_str = str(race_id)
        if race_id_str not in payout_map:
            continue
        
        sorted_group = group.sort_values('pred_rank')
        if len(sorted_group) < 5:
            continue
        
        # Get month
        month = sorted_group.iloc[0]['date'].month
        if month not in monthly_stats:
            monthly_stats[month] = {'cost': 0, 'return': 0, 'races': 0, 'hits': 0, 
                                     'high_ev_count': 0, 'mid_ev_count': 0, 'skip_count': 0}
        
        top1 = sorted_group.iloc[0]
        
        # Calculate EV using expected_value or softmax probability * odds
        if 'expected_value' in sorted_group.columns and pd.notna(top1.get('expected_value')):
            top1_ev = float(top1.get('expected_value', 0))
        else:
            probs = softmax(sorted_group['score'].values)
            top1_prob = probs[0]
            top1_odds = float(top1.get('odds', 1)) if pd.notna(top1.get('odds')) else 1.0
            top1_ev = top1_prob * top1_odds
        
        # Prepare horses
        axis = int(sorted_group.iloc[0]['horse_number'])
        opps = [int(sorted_group.iloc[i]['horse_number']) for i in range(1, min(5, len(sorted_group)))]
        
        race_cost = 0
        race_return = 0
        hit = False
        
        if top1_ev >= 1.2:
            # High Value: 三連複 Top1→3頭 (3点)
            monthly_stats[month]['high_ev_count'] += 1
            tickets = list(combinations([axis] + opps[:3], 3))
            tickets = [t for t in tickets if axis in t]
            race_cost = len(tickets) * 100
            
            for t in tickets:
                sorted_t = tuple(sorted(t))
                key = f"{sorted_t[0]:02}{sorted_t[1]:02}{sorted_t[2]:02}"
                if key in payout_map[race_id_str].get('sanrenpuku', {}):
                    race_return += payout_map[race_id_str]['sanrenpuku'][key]
                    hit = True
                    
        elif top1_ev >= 0.8:
            # Mid Value: 三連単 Top1→3頭 (6点)
            monthly_stats[month]['mid_ev_count'] += 1
            tickets = [(axis, o1, o2) for o1, o2 in permutations(opps[:3], 2)]
            race_cost = len(tickets) * 100
            
            for t in tickets:
                key = f"{t[0]:02}{t[1]:02}{t[2]:02}"
                if key in payout_map[race_id_str].get('sanrentan', {}):
                    race_return += payout_map[race_id_str]['sanrentan'][key]
                    hit = True
        else:
            # Low Value: Skip
            monthly_stats[month]['skip_count'] += 1
        
        monthly_stats[month]['cost'] += race_cost
        monthly_stats[month]['return'] += race_return
        monthly_stats[month]['races'] += 1
        if hit:
            monthly_stats[month]['hits'] += 1
    
    return monthly_stats

def print_report(monthly_stats):
    """Print formatted monthly report."""
    print("\n" + "="*80)
    print("v12 EV Strategy - 2025 Monthly ROI Report")
    print("="*80)
    print(f"{'月':<5} {'投資額':>12} {'払戻額':>12} {'収支':>12} {'回収率':>8} {'的中':>6} {'High':>6} {'Mid':>6} {'Skip':>6}")
    print("-"*80)
    
    total_cost = 0
    total_return = 0
    total_hits = 0
    
    for month in sorted(monthly_stats.keys()):
        stats = monthly_stats[month]
        cost = stats['cost']
        ret = stats['return']
        profit = ret - cost
        roi = (ret / cost * 100) if cost > 0 else 0
        
        total_cost += cost
        total_return += ret
        total_hits += stats['hits']
        
        print(f"{month:>2}月  {cost:>10,}円 {ret:>10,}円 {profit:>+10,}円 {roi:>7.1f}% {stats['hits']:>5} {stats['high_ev_count']:>5} {stats['mid_ev_count']:>5} {stats['skip_count']:>5}")
    
    print("-"*80)
    total_profit = total_return - total_cost
    total_roi = (total_return / total_cost * 100) if total_cost > 0 else 0
    print(f"{'合計':<4} {total_cost:>10,}円 {total_return:>10,}円 {total_profit:>+10,}円 {total_roi:>7.1f}% {total_hits:>5}")
    print("="*80)

if __name__ == "__main__":
    try:
        print("Loading data...")
        df, payout_df = load_data()
        
        print("Building payout map...")
        payout_map = build_payout_map(payout_df)
        
        print("Calculating monthly ROI...")
        monthly_stats = calculate_ev_strategy(df, payout_map, year=2025)
        
        print_report(monthly_stats)
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()

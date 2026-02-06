"""
Calculate monthly ROI using Win (単勝), Place (複勝), Umaren (馬連) combination
Using PayoutLoader for actual payout data
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pandas as pd
import numpy as np
from collections import defaultdict

print("Loading modules...")

from utils.payout_loader import PayoutLoader

# Load OOF data
print("Loading OOF data...")
oof = pd.read_parquet('data/predictions/v13_wf_2025_full_retrained_oof_calibrated.parquet')

# Load preprocessed data for date
print("Loading preprocessed data...")
prep = pd.read_parquet('data/processed/preprocessed_data.parquet', 
                       columns=['race_id', 'horse_number', 'date'])
prep['race_id'] = prep['race_id'].astype(str)
prep = prep.dropna(subset=['horse_number'])
prep['horse_number'] = prep['horse_number'].astype(float).astype(int)

# Merge
oof['race_id'] = oof['race_id'].astype(str)
oof = oof.dropna(subset=['horse_number'])
oof['horse_number'] = oof['horse_number'].astype(float).astype(int)
df = pd.merge(oof, prep[['race_id', 'horse_number', 'date']], on=['race_id', 'horse_number'], how='left')

# Filter valid data
df = df.dropna(subset=['rank', 'calib_prob_iso', 'date'])
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month

# Strategy parameters
BET_UNIT = 100  # ¥100 per ticket
SLIPPAGE_FACTOR = 0.90
INITIAL_BANKROLL = 1000000  # 100万円
MAX_BET_PER_RACE = 10000    # 1万円

# Load payout data
print("Loading payout data...")
loader = PayoutLoader()
payout_map = loader.load_payout_map([2025])

print(f"Strategy: Win + Place + Umaren (Top2)")
print(f"Bet Unit: ¥{BET_UNIT}, Slippage: {SLIPPAGE_FACTOR}")
print(f"Bankroll: ¥{INITIAL_BANKROLL:,}, Max per race: ¥{MAX_BET_PER_RACE:,}")
print("")

def format_win(horse):
    """Format for tansho/fukusho"""
    return f"{horse:02}"

def format_umaren(horse1, horse2):
    """Format for umaren (sorted, 2-digit each)"""
    return "".join([f"{h:02}" for h in sorted([horse1, horse2])])

# Stats per month
results = []

for month in sorted(df['month'].unique()):
    month_df = df[df['month'] == month]
    
    month_stats = {
        'month': month, 
        'total_stake': 0, 
        'total_payout': 0,
        'races': 0,
        'win_hits': 0,
        'place_hits': 0,
        'umaren_hits': 0
    }
    
    for race_id in sorted(month_df['race_id'].unique()):
        grp = month_df[month_df['race_id'] == race_id].copy()
        
        if len(grp) < 2:
            continue
        
        month_stats['races'] += 1
        
        # Get top 2 horses by model probability
        grp = grp.sort_values('calib_prob_iso', ascending=False)
        top1 = int(grp.iloc[0]['horse_number'])
        top2 = int(grp.iloc[1]['horse_number'])
        
        # Check payouts
        if race_id not in payout_map:
            continue
        
        race_payouts = payout_map[race_id]
        
        # === WIN (単勝) - Top1 ===
        win_key = format_win(top1)
        tansho_payouts = race_payouts.get('tansho', {})
        
        month_stats['total_stake'] += BET_UNIT
        if win_key in tansho_payouts:
            payout = tansho_payouts[win_key] * SLIPPAGE_FACTOR * (BET_UNIT / 100)
            month_stats['total_payout'] += payout
            month_stats['win_hits'] += 1
        
        # === PLACE (複勝) - Top1 ===
        fukusho_payouts = race_payouts.get('fukusho', {})
        
        month_stats['total_stake'] += BET_UNIT
        if win_key in fukusho_payouts:
            payout = fukusho_payouts[win_key] * SLIPPAGE_FACTOR * (BET_UNIT / 100)
            month_stats['total_payout'] += payout
            month_stats['place_hits'] += 1
        
        # === UMAREN (馬連) - Top1-Top2 ===
        umaren_key = format_umaren(top1, top2)
        umaren_payouts = race_payouts.get('umaren', {})
        
        month_stats['total_stake'] += BET_UNIT
        if umaren_key in umaren_payouts:
            payout = umaren_payouts[umaren_key] * SLIPPAGE_FACTOR * (BET_UNIT / 100)
            month_stats['total_payout'] += payout
            month_stats['umaren_hits'] += 1
    
    # Calc ROI
    month_stats['roi'] = (month_stats['total_payout'] / month_stats['total_stake'] * 100 
                          if month_stats['total_stake'] > 0 else 0)
    
    results.append(month_stats)

# Print table
print()
print('| 月 | レース数 | 単勝的中 | 複勝的中 | 馬連的中 | 投資額 | 回収額 | ROI |')
print('| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |')
total_stake, total_payout, total_races = 0, 0, 0
total_win, total_place, total_umaren = 0, 0, 0
for r in results:
    print(f"| {r['month']}月 | {r['races']} | {r['win_hits']} | {r['place_hits']} | {r['umaren_hits']} | {r['total_stake']:,.0f}円 | {r['total_payout']:,.0f}円 | {r['roi']:.1f}% |")
    total_stake += r['total_stake']
    total_payout += r['total_payout']
    total_races += r['races']
    total_win += r['win_hits']
    total_place += r['place_hits']
    total_umaren += r['umaren_hits']

total_roi = total_payout / total_stake * 100 if total_stake > 0 else 0
print(f'| **合計** | **{total_races}** | **{total_win}** | **{total_place}** | **{total_umaren}** | **{total_stake:,.0f}円** | **{total_payout:,.0f}円** | **{total_roi:.1f}%** |')

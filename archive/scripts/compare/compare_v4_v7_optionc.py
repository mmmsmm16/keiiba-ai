"""v4 vs v7 Option C 比較 (修正版)"""
import pandas as pd
import numpy as np
from itertools import permutations

v4 = pd.read_parquet('experiments/predictions_ensemble_v4_2025.parquet')
v7 = pd.read_parquet('experiments/v7_ensemble_full/reports/predictions.parquet')

# Filter v4 to JRA only
jra_codes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
v4['venue_code'] = v4['venue'].astype(str).str[:2]
v4 = v4[v4['venue_code'].isin(jra_codes)].copy()

# Load payouts
payout_df = pd.read_parquet('experiments/payouts_2024_2025.parquet')
payout_df = payout_df[payout_df['race_id'].str[:4] == '2025']

payout_map = {}
for _, row in payout_df.iterrows():
    rid = row['race_id']
    payout_map[rid] = {'sanrentan': {}, 'tansho': {}}
    for i in range(1, 7):
        col_a = f'haraimodoshi_sanrentan_{i}a'
        col_b = f'haraimodoshi_sanrentan_{i}b'
        if col_a in row and pd.notna(row.get(col_a)):
            try: 
                payout_map[rid]['sanrentan'][str(row[col_a]).strip()] = int(float(str(row[col_b]).strip()))
            except: pass
    col_a, col_b = 'haraimodoshi_tansho_1a', 'haraimodoshi_tansho_1b'
    if col_a in row and pd.notna(row.get(col_a)):
        try: 
            payout_map[rid]['tansho'][str(int(row[col_a])).zfill(2)] = int(float(str(row[col_b]).strip()))
        except: pass

def run_strategy(pred_df, name):
    total = {'cost': 0, 'return': 0, 'races': 0}
    for rid, grp in pred_df.groupby('race_id'):
        if rid not in payout_map: continue
        sorted_g = grp.sort_values('score', ascending=False)
        if len(sorted_g) < 6: continue
        
        top1 = sorted_g.iloc[0]
        h = sorted_g['horse_number'].astype(int).tolist()
        pop = int(top1['popularity']) if pd.notna(top1.get('popularity', np.nan)) else 99
        scores = sorted_g.head(6)['score'].values
        gap = float(scores[0] - scores[5]) if len(scores) >= 6 else 0.5
        
        if pop >= 7:
            tickets = [(h[0], o1, o2) for o1, o2 in permutations(h[1:4], 2)]
            cost = len(tickets) * 100
            ret = sum(payout_map[rid]['sanrentan'].get(f'{t[0]:02}{t[1]:02}{t[2]:02}', 0) for t in tickets)
        elif gap < 0.3:
            tickets = [(h[0], o1, o2) for o1, o2 in permutations(h[1:4], 2)]
            cost = len(tickets) * 100
            ret = sum(payout_map[rid]['sanrentan'].get(f'{t[0]:02}{t[1]:02}{t[2]:02}', 0) for t in tickets)
        else:
            cost = 100
            ret = payout_map[rid]['tansho'].get(f'{h[0]:02}', 0)
        
        total['cost'] += cost
        total['return'] += ret
        total['races'] += 1
    
    roi = total['return']/total['cost']*100 if total['cost'] > 0 else 0
    profit = total['return'] - total['cost']
    print(f"{name}: {total['races']}レース, 投資{total['cost']:,}円, 回収{total['return']:,}円, 利益{profit:+,}円, ROI {roi:.1f}%")
    return roi, profit

print('=== Option C (B+C戦略) v4 vs v7 比較 ===')
print()
v4_roi, v4_profit = run_strategy(v4, 'v4_2025')
v7_roi, v7_profit = run_strategy(v7, 'v7     ')
print()
print(f"差: ROI {v7_roi - v4_roi:+.1f}%, 利益 {v7_profit - v4_profit:+,}円")

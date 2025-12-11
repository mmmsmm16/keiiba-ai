"""最適化適応型戦略（券種分岐あり）"""
import pandas as pd
import numpy as np
from itertools import combinations, permutations

# Load data
pred = pd.read_parquet('experiments/v7_ensemble_full/reports/predictions.parquet')
payout_df = pd.read_parquet('experiments/payouts_2024_2025.parquet')
payout_df = payout_df[payout_df['race_id'].str[:4] == '2025']

# Build payout map
payout_map = {}
for _, row in payout_df.iterrows():
    rid = row['race_id']
    payout_map[rid] = {'sanrentan': {}, 'umaren': {}, 'sanrenpuku': {}}
    for i in range(1, 7):
        col_a, col_b = f'haraimodoshi_sanrentan_{i}a', f'haraimodoshi_sanrentan_{i}b'
        if col_a in row and pd.notna(row.get(col_a)):
            try: 
                payout_map[rid]['sanrentan'][str(row[col_a]).strip()] = int(float(str(row[col_b]).strip()))
            except: pass
    for i in range(1, 4):
        col_a, col_b = f'haraimodoshi_umaren_{i}a', f'haraimodoshi_umaren_{i}b'
        if col_a in row and pd.notna(row.get(col_a)):
            try: 
                payout_map[rid]['umaren'][str(row[col_a]).strip()] = int(float(str(row[col_b]).strip()))
            except: pass
        col_a, col_b = f'haraimodoshi_sanrenpuku_{i}a', f'haraimodoshi_sanrenpuku_{i}b'
        if col_a in row and pd.notna(row.get(col_a)):
            try: 
                payout_map[rid]['sanrenpuku'][str(row[col_a]).strip()] = int(float(str(row[col_b]).strip()))
            except: pass

print(f"Loaded: {len(pred)} rows, payout: {len(payout_map)} races")

# Prepare race data
race_data = {}
for rid, grp in pred.groupby('race_id'):
    sorted_g = grp.sort_values('score', ascending=False)
    if len(sorted_g) < 6: continue
    top1 = sorted_g.iloc[0]
    scores = sorted_g.head(6)['score'].values
    race_data[rid] = {
        'horses': sorted_g['horse_number'].astype(int).tolist(),
        'top1_popularity': int(top1['popularity']) if pd.notna(top1.get('popularity', np.nan)) else 99,
        'score_range': scores[0] - scores[5]
    }

# Strategy functions
def sim_sanrentan(h, n, pmap, rid):
    if len(h) < n: return 0, 0
    axis, opps = h[0], h[1:n]
    tickets = [(axis, o1, o2) for o1, o2 in permutations(opps, 2)]
    cost = len(tickets) * 100
    ret = sum(pmap.get(rid,{}).get('sanrentan',{}).get(f'{t[0]:02}{t[1]:02}{t[2]:02}',0) for t in tickets)
    return cost, ret

def sim_sanrenpuku(h, n, pmap, rid):
    if len(h) < n: return 0, 0
    tickets = list(combinations(h[:n], 3))
    cost = len(tickets) * 100
    ret = 0
    for t in tickets:
        ts = sorted(t)
        key = f'{ts[0]:02}{ts[1]:02}{ts[2]:02}'
        ret += pmap.get(rid,{}).get('sanrenpuku',{}).get(key,0)
    return cost, ret

def sim_umaren(h, n, pmap, rid):
    if len(h) < n: return 0, 0
    tickets = list(combinations(h[:n], 2))
    cost = len(tickets) * 100
    ret = 0
    for t in tickets:
        ts = sorted(t)
        key = f'{ts[0]:02}{ts[1]:02}'
        ret += pmap.get(rid,{}).get('umaren',{}).get(key,0)
    return cost, ret

# Optimized Adaptive Strategy
total_cost = 0
total_return = 0
strategy_stats = {}

for rid, rd in race_data.items():
    if rid not in payout_map: continue
    pop, gap, h = rd['top1_popularity'], rd['score_range'], rd['horses']
    
    if pop >= 7:
        strategy = '7番人気以上: 三連単1軸4頭'
        cost, ret = sim_sanrentan(h, 4, payout_map, rid)
    elif gap < 0.3:
        strategy = '接戦: 三連単1軸4頭'
        cost, ret = sim_sanrentan(h, 4, payout_map, rid)
    elif 4 <= pop <= 6:
        # 4-6番人気 -> 三連複Box5に変更（過去グリッドサーチで効いていた）
        strategy = '4-6番人気: 三連複Box5'
        cost, ret = sim_sanrenpuku(h, 5, payout_map, rid)
    else:
        # その他(1-3番人気+gap>=0.3) -> 三連複Box5頭
        strategy = 'その他: 三連複Box5'
        cost, ret = sim_sanrenpuku(h, 5, payout_map, rid)
    
    if cost > 0:
        total_cost += cost
        total_return += ret
        if strategy not in strategy_stats:
            strategy_stats[strategy] = {'cost': 0, 'return': 0, 'count': 0}
        strategy_stats[strategy]['cost'] += cost
        strategy_stats[strategy]['return'] += ret
        strategy_stats[strategy]['count'] += 1

roi = total_return / total_cost * 100 if total_cost > 0 else 0
total_races = sum(s['count'] for s in strategy_stats.values())

print()
print('=' * 70)
print('=== 最適化適応型戦略（券種分岐あり）===')
print('=' * 70)
print(f"総レース数: {total_races}")
print(f"投資総額: {total_cost:,}円")
print(f"回収総額: {total_return:,}円")
print(f"利益: {total_return - total_cost:+,}円")
print(f"ROI: {roi:.1f}%")
print()
print('--- 戦略別内訳 ---')
for s, st in sorted(strategy_stats.items(), key=lambda x: -x[1]['return']/x[1]['cost'] if x[1]['cost']>0 else 0):
    s_roi = st['return']/st['cost']*100 if st['cost']>0 else 0
    print(f"{s}: {st['count']}回, 投資{st['cost']:,}円, 回収{st['return']:,}円, ROI {s_roi:.1f}%")

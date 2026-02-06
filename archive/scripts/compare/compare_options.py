"""3つの戦略オプションを比較"""
import pandas as pd
import numpy as np
from itertools import combinations, permutations

pred = pd.read_parquet('experiments/v7_ensemble_full/reports/predictions.parquet')
payout_df = pd.read_parquet('experiments/payouts_2024_2025.parquet')
payout_df = payout_df[payout_df['race_id'].str[:4] == '2025']

payout_map = {}
for _, row in payout_df.iterrows():
    rid = row['race_id']
    payout_map[rid] = {'sanrentan': {}, 'sanrenpuku': {}, 'tansho': {}}
    for i in range(1, 7):
        col_a, col_b = f'haraimodoshi_sanrentan_{i}a', f'haraimodoshi_sanrentan_{i}b'
        if col_a in row and pd.notna(row.get(col_a)):
            try: payout_map[rid]['sanrentan'][str(row[col_a]).strip()] = int(float(str(row[col_b]).strip()))
            except: pass
    for i in range(1, 4):
        col_a, col_b = f'haraimodoshi_sanrenpuku_{i}a', f'haraimodoshi_sanrenpuku_{i}b'
        if col_a in row and pd.notna(row.get(col_a)):
            try: payout_map[rid]['sanrenpuku'][str(row[col_a]).strip()] = int(float(str(row[col_b]).strip()))
            except: pass
    # 単勝
    col_a, col_b = 'haraimodoshi_tansho_1a', 'haraimodoshi_tansho_1b'
    if col_a in row and pd.notna(row.get(col_a)):
        try: payout_map[rid]['tansho'][str(int(row[col_a])).zfill(2)] = int(float(str(row[col_b]).strip()))
        except: pass

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

def sim_sanrentan(h, n, pmap, rid):
    if len(h) < n: return 0, 0
    tickets = [(h[0], o1, o2) for o1, o2 in permutations(h[1:n], 2)]
    cost = len(tickets) * 100
    ret = sum(pmap.get(rid,{}).get('sanrentan',{}).get(f'{t[0]:02}{t[1]:02}{t[2]:02}',0) for t in tickets)
    return cost, ret

def sim_sanrenpuku(h, n, pmap, rid):
    if len(h) < n: return 0, 0
    tickets = list(combinations(h[:n], 3))
    cost = len(tickets) * 100
    ret = sum(pmap.get(rid,{}).get('sanrenpuku',{}).get(''.join(f'{x:02}' for x in sorted(t)),0) for t in tickets)
    return cost, ret

def sim_tansho(h, pmap, rid):
    cost = 100
    horse_key = f'{h[0]:02}'
    ret = pmap.get(rid,{}).get('tansho',{}).get(horse_key, 0)
    return cost, ret

print('=' * 70)
print('=== 3つの戦略オプション比較 ===')
print('=' * 70)

# Option A: 4-6番人気を「その他」に統合（全て三連複）
total_a = {'cost': 0, 'return': 0, 'races': 0}
for rid, rd in race_data.items():
    if rid not in payout_map: continue
    pop, gap, h = rd['top1_popularity'], rd['score_range'], rd['horses']
    if pop >= 7:
        cost, ret = sim_sanrentan(h, 4, payout_map, rid)
    elif gap < 0.3:
        cost, ret = sim_sanrentan(h, 4, payout_map, rid)
    else:
        cost, ret = sim_sanrenpuku(h, 5, payout_map, rid)
    if cost > 0:
        total_a['cost'] += cost
        total_a['return'] += ret
        total_a['races'] += 1
roi_a = total_a['return']/total_a['cost']*100
profit_a = total_a['return'] - total_a['cost']
print()
print('Option A: 4-6番人気を「その他」統合 (三連複Box5)')
print(f"  レース: {total_a['races']}, 投資: {total_a['cost']:,}円, 回収: {total_a['return']:,}円")
print(f"  利益: {profit_a:+,}円, ROI: {roi_a:.1f}%")

# Option B: ROI 100%超の条件のみ（7番人気以上+接戦のみ）
total_b = {'cost': 0, 'return': 0, 'races': 0}
for rid, rd in race_data.items():
    if rid not in payout_map: continue
    pop, gap, h = rd['top1_popularity'], rd['score_range'], rd['horses']
    if pop >= 7:
        cost, ret = sim_sanrentan(h, 4, payout_map, rid)
    elif gap < 0.3:
        cost, ret = sim_sanrentan(h, 4, payout_map, rid)
    else:
        continue
    if cost > 0:
        total_b['cost'] += cost
        total_b['return'] += ret
        total_b['races'] += 1
roi_b = total_b['return']/total_b['cost']*100
profit_b = total_b['return'] - total_b['cost']
print()
print('Option B: ROI 100%超の条件のみ (7番人気以上+接戦)')
print(f"  レース: {total_b['races']}, 投資: {total_b['cost']:,}円, 回収: {total_b['return']:,}円")
print(f"  利益: {profit_b:+,}円, ROI: {roi_b:.1f}%")

# Option C: その他は単勝
total_c = {'cost': 0, 'return': 0, 'races': 0}
for rid, rd in race_data.items():
    if rid not in payout_map: continue
    pop, gap, h = rd['top1_popularity'], rd['score_range'], rd['horses']
    if pop >= 7:
        cost, ret = sim_sanrentan(h, 4, payout_map, rid)
    elif gap < 0.3:
        cost, ret = sim_sanrentan(h, 4, payout_map, rid)
    else:
        cost, ret = sim_tansho(h, payout_map, rid)
    if cost > 0:
        total_c['cost'] += cost
        total_c['return'] += ret
        total_c['races'] += 1
roi_c = total_c['return']/total_c['cost']*100
profit_c = total_c['return'] - total_c['cost']
print()
print('Option C: その他は単勝')
print(f"  レース: {total_c['races']}, 投資: {total_c['cost']:,}円, 回収: {total_c['return']:,}円")
print(f"  利益: {profit_c:+,}円, ROI: {roi_c:.1f}%")

print()
print('=' * 70)
print('=== 比較サマリー ===')
print('=' * 70)
print(f"{'オプション':<40} | {'ROI':>8} | {'利益':>15}")
print('-' * 70)
print(f"{'A: 4-6統合→三連複Box5':<40} | {roi_a:>7.1f}% | {profit_a:>+14,}円")
print(f"{'B: ROI100%超のみ賭け(7番+接戦)':<40} | {roi_b:>7.1f}% | {profit_b:>+14,}円")
print(f"{'C: その他→単勝':<40} | {roi_c:>7.1f}% | {profit_c:>+14,}円")
print('=' * 70)

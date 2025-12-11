"""
拡張 Out-of-Sample Backtest: 2024年グリッドサーチの全戦略を2025年v7でテスト

2024年で見つかった戦略（ROI 80%以上含む）を全て2025年に適用
"""
import pandas as pd
import numpy as np
from itertools import combinations, permutations
from collections import defaultdict

# --- Load v7 2025 predictions (JRA only) ---
print("Loading v7 predictions...")
pred = pd.read_parquet('experiments/v7_ensemble_full/reports/predictions.parquet')
print(f"Loaded: {len(pred)} rows, {pred['race_id'].nunique()} races")

# --- Load payouts ---
print("Loading payouts...")
payout_df = pd.read_parquet('experiments/payouts_2024_2025.parquet')
payout_df = payout_df[payout_df['race_id'].str[:4] == '2025']
print(f"Payout data: {len(payout_df)} races")

# Build payout map
payout_map = {}
for _, row in payout_df.iterrows():
    rid = row['race_id']
    payout_map[rid] = {'sanrentan': {}, 'umaren': {}, 'sanrenpuku': {}}
    
    for i in range(1, 7):
        col_a = f'haraimodoshi_sanrentan_{i}a'
        col_b = f'haraimodoshi_sanrentan_{i}b'
        if col_a in row and pd.notna(row.get(col_a)):
            try:
                key = str(row[col_a]).strip()
                val = int(float(str(row[col_b]).strip()))
                payout_map[rid]['sanrentan'][key] = val
            except: pass
    
    for i in range(1, 4):
        col_a = f'haraimodoshi_umaren_{i}a'
        col_b = f'haraimodoshi_umaren_{i}b'
        if col_a in row and pd.notna(row.get(col_a)):
            try:
                key = str(row[col_a]).strip()
                val = int(float(str(row[col_b]).strip()))
                payout_map[rid]['umaren'][key] = val
            except: pass
    
    for i in range(1, 4):
        col_a = f'haraimodoshi_sanrenpuku_{i}a'
        col_b = f'haraimodoshi_sanrenpuku_{i}b'
        if col_a in row and pd.notna(row.get(col_a)):
            try:
                key = str(row[col_a]).strip()
                val = int(float(str(row[col_b]).strip()))
                payout_map[rid]['sanrenpuku'][key] = val
            except: pass

print(f"Payout map: {len(payout_map)} races")

# --- Prepare race data ---
race_data = {}
for rid, grp in pred.groupby('race_id'):
    sorted_g = grp.sort_values('score', ascending=False)
    if len(sorted_g) < 6:
        continue
    
    top1 = sorted_g.iloc[0]
    scores = sorted_g.head(6)['score'].values
    score_range = scores[0] - scores[5] if len(scores) >= 6 else 0
    
    race_data[rid] = {
        'horses': sorted_g['horse_number'].astype(int).tolist(),
        'top1_popularity': int(top1['popularity']) if pd.notna(top1.get('popularity', np.nan)) else 99,
        'top1_odds': float(top1['odds']) if pd.notna(top1.get('odds', np.nan)) else 0,
        'score_range': score_range
    }

print(f"Prepared: {len(race_data)} races")

# --- Simulation functions ---
def simulate_sanrentan(rd, rid, formation_type):
    h = rd['horses']
    tickets = []
    
    if formation_type.startswith('box_'):
        n = int(formation_type.split('_')[1])
        if len(h) < n: return 0, 0, 0
        top_n = h[:n]
        tickets = list(permutations(top_n, 3))
        
    elif formation_type.startswith('nagashi_1_'):
        n = int(formation_type.split('_')[2])
        if len(h) < n: return 0, 0, 0
        axis = h[0]
        opps = h[1:n]
        tickets = [(axis, o1, o2) for o1, o2 in permutations(opps, 2)]
        
    elif formation_type.startswith('nagashi_2_'):
        n = int(formation_type.split('_')[2])
        if len(h) < n: return 0, 0, 0
        axis1, axis2 = h[0], h[1]
        opps = h[2:n]
        tickets += [(axis1, axis2, o) for o in opps]
        tickets += [(axis1, o, axis2) for o in opps]
        for o1, o2 in permutations(opps, 2):
            tickets.append((axis1, o1, o2))
        tickets += [(axis2, axis1, o) for o in opps]
        tickets += [(axis2, o, axis1) for o in opps]
        for o1, o2 in permutations(opps, 2):
            tickets.append((axis2, o1, o2))
        tickets = list(set(tickets))
    
    if not tickets: return 0, 0, 0
    
    cost = len(tickets) * 100
    ret = 0
    hit = 0
    
    for t in tickets:
        key = f"{t[0]:02}{t[1]:02}{t[2]:02}"
        if key in payout_map.get(rid, {}).get('sanrentan', {}):
            ret += payout_map[rid]['sanrentan'][key]
            hit = 1
    
    return cost, ret, hit

def simulate_umaren(rd, rid, n_heads):
    h = rd['horses']
    if len(h) < n_heads: return 0, 0, 0
    
    top_n = h[:n_heads]
    tickets = list(combinations(top_n, 2))
    cost = len(tickets) * 100
    ret = 0
    hit = 0
    
    for t in tickets:
        t_sorted = sorted(t)
        key = f"{t_sorted[0]:02}{t_sorted[1]:02}"
        if key in payout_map.get(rid, {}).get('umaren', {}):
            ret += payout_map[rid]['umaren'][key]
            hit = 1
    
    return cost, ret, hit

def simulate_sanrenpuku(rd, rid):
    h = rd['horses']
    if len(h) < 5: return 0, 0, 0
    
    top5 = h[:5]
    tickets = list(combinations(top5, 3))
    cost = len(tickets) * 100
    ret = 0
    hit = 0
    
    for t in tickets:
        t_sorted = sorted(t)
        key = f"{t_sorted[0]:02}{t_sorted[1]:02}{t_sorted[2]:02}"
        if key in payout_map.get(rid, {}).get('sanrenpuku', {}):
            ret += payout_map[rid]['sanrenpuku'][key]
            hit = 1
    
    return cost, ret, hit

def run_backtest(bet_type, formation=None, condition_func=None, condition_name="全体"):
    stats = {'races': 0, 'cost': 0, 'return': 0, 'hits': 0}
    
    for rid, rd in race_data.items():
        if rid not in payout_map: continue
        if condition_func and not condition_func(rd): continue
        
        if bet_type == 'sanrentan':
            cost, ret, hit = simulate_sanrentan(rd, rid, formation)
        elif bet_type == 'umaren':
            cost, ret, hit = simulate_umaren(rd, rid, int(formation))
        elif bet_type == 'sanrenpuku':
            cost, ret, hit = simulate_sanrenpuku(rd, rid)
        else:
            continue
            
        if cost > 0:
            stats['races'] += 1
            stats['cost'] += cost
            stats['return'] += ret
            stats['hits'] += hit
    
    roi = stats['return'] / stats['cost'] * 100 if stats['cost'] > 0 else 0
    hit_rate = stats['hits'] / stats['races'] * 100 if stats['races'] > 0 else 0
    
    return {
        'roi': roi,
        'races': stats['races'],
        'cost': stats['cost'],
        'return': stats['return'],
        'hit_rate': hit_rate
    }

# --- Run Extended Backtest ---
print()
print('=' * 80)
print('=== 拡張 OUT-OF-SAMPLE BACKTEST: 2024年全戦略 → 2025年v7適用 ===')
print('=' * 80)

# 条件関数
conditions = [
    (None, '全体', None),
    (lambda rd: rd['top1_popularity'] == 1, '1番人気', 'pop_1'),
    (lambda rd: 2 <= rd['top1_popularity'] <= 3, '2-3番人気', 'pop_23'),
    (lambda rd: 4 <= rd['top1_popularity'] <= 6, '4-6番人気', 'pop_46'),
    (lambda rd: rd['top1_popularity'] >= 7, '7番人気以上', 'pop_7plus'),
    (lambda rd: rd['score_range'] < 0.3, '接戦(gap<0.3)', 'gap_small'),
    (lambda rd: 0.3 <= rd['score_range'] < 0.6, '中差(0.3-0.6)', 'gap_medium'),
    (lambda rd: rd['score_range'] >= 0.6, '大差(gap>=0.6)', 'gap_large'),
]

# フォーメーション
formations_sanrentan = [
    ('nagashi_1_4', '1頭軸→相手4頭', 6),
    ('nagashi_1_5', '1頭軸→相手5頭', 12),
    ('nagashi_1_6', '1頭軸→相手6頭', 20),
    ('nagashi_1_7', '1頭軸→相手7頭', 30),
    ('nagashi_2_4', '2頭軸→相手4頭', 12),
    ('nagashi_2_5', '2頭軸→相手5頭', None),
    ('box_3', 'Box3頭', 6),
    ('box_4', 'Box4頭', 24),
    ('box_5', 'Box5頭', 60),
]

results = []

# 三連単
print("\n--- 三連単 ---")
for form_id, form_name, points in formations_sanrentan:
    for cond_func, cond_name, cond_id in conditions:
        result = run_backtest('sanrentan', form_id, cond_func, cond_name)
        if result['races'] >= 20:  # 最低20レース以上
            results.append({
                'bet_type': '三連単',
                'formation': form_name,
                'condition': cond_name,
                'roi': result['roi'],
                'races': result['races'],
                'hit_rate': result['hit_rate'],
                'points': points
            })

# 馬連
print("\n--- 馬連 ---")
for n in [3, 4, 5, 6]:
    for cond_func, cond_name, cond_id in conditions:
        result = run_backtest('umaren', str(n), cond_func, cond_name)
        if result['races'] >= 20:
            results.append({
                'bet_type': '馬連',
                'formation': f'Box{n}頭',
                'condition': cond_name,
                'roi': result['roi'],
                'races': result['races'],
                'hit_rate': result['hit_rate'],
                'points': n * (n-1) // 2
            })

# 三連複
print("\n--- 三連複 ---")
for cond_func, cond_name, cond_id in conditions:
    result = run_backtest('sanrenpuku', None, cond_func, cond_name)
    if result['races'] >= 20:
        results.append({
            'bet_type': '三連複',
            'formation': 'Box5頭',
            'condition': cond_name,
            'roi': result['roi'],
            'races': result['races'],
            'hit_rate': result['hit_rate'],
            'points': 10
        })

# ソート・表示
results = sorted(results, key=lambda x: x['roi'], reverse=True)

print()
print('=' * 90)
print(f"{'券種':<8} | {'フォーメーション':<18} | {'条件':<15} | {'ROI':>8} | {'レース':>6} | {'的中率':>6} | {'点数':>4}")
print('-' * 90)

for r in results[:40]:
    roi_mark = "🏆" if r['roi'] >= 100 else ""
    print(f"{r['bet_type']:<8} | {r['formation']:<18} | {r['condition']:<15} | {r['roi']:>7.1f}% | {r['races']:>6} | {r['hit_rate']:>5.1f}% | {r['points'] or 'N/A':>4} {roi_mark}")

# ROI 100%超
over_100 = [r for r in results if r['roi'] >= 100]
print()
print('=' * 90)
print(f"🏆 ROI 100%以上の戦略: {len(over_100)}件")
print('=' * 90)
for r in over_100:
    print(f"  {r['bet_type']} {r['formation']} × {r['condition']}: ROI {r['roi']:.1f}% ({r['races']}レース, 的中率{r['hit_rate']:.1f}%)")

# ROI 80-100%
between_80_100 = [r for r in results if 80 <= r['roi'] < 100]
print()
print(f"📊 ROI 80-100%の戦略: {len(between_80_100)}件 (上位10)")
for r in between_80_100[:10]:
    print(f"  {r['bet_type']} {r['formation']} × {r['condition']}: ROI {r['roi']:.1f}% ({r['races']}レース)")

print()
print('=' * 90)

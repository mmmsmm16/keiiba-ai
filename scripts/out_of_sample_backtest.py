"""
Out-of-Sample Backtest: 過去(2024年)でグリッドサーチした戦略を2025年v7に適用

目的: 2024年で最適化された戦略パラメータを、2025年の新データに適用し、
      真のout-of-sample検証を行う
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

def run_backtest(formation_type, condition_func=None, condition_name="全体"):
    stats = {'races': 0, 'cost': 0, 'return': 0, 'hits': 0}
    
    for rid, rd in race_data.items():
        if rid not in payout_map: continue
        if condition_func and not condition_func(rd): continue
        
        cost, ret, hit = simulate_sanrentan(rd, rid, formation_type)
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

# --- Run Out-of-Sample Backtest with strategies from 2024 grid search ---
print()
print('=' * 70)
print('=== OUT-OF-SAMPLE BACKTEST: 2024年最適化戦略 → 2025年v7適用 ===')
print('=' * 70)

# 条件関数
pop_7plus = lambda rd: rd['top1_popularity'] >= 7
pop_4_6 = lambda rd: 4 <= rd['top1_popularity'] <= 6
gap_small = lambda rd: rd['score_range'] < 0.3

# 過去グリッドサーチでROI 100%超だった戦略をテスト
tests = [
    # (formation, condition_func, condition_name, past_roi)
    ('nagashi_1_4', pop_7plus, '7番人気以上', 232.8),
    ('nagashi_2_4', pop_7plus, '7番人気以上', 174.0),
    ('nagashi_1_6', pop_4_6, '4-6番人気', 123.9),
    ('nagashi_1_7', pop_4_6, '4-6番人気', 119.5),
    ('box_5', pop_7plus, '7番人気以上', 118.1),
    ('nagashi_1_5', pop_7plus, '7番人気以上', 116.4),
    ('box_4', pop_7plus, '7番人気以上', 109.0),
    ('nagashi_1_6', gap_small, '接戦(gap<0.3)', 106.1),
    ('nagashi_1_7', pop_7plus, '7番人気以上', 105.8),
    ('nagashi_2_4', gap_small, '接戦(gap<0.3)', 100.3),
    # 無条件テストも追加
    ('nagashi_1_6', None, '全体', 88.9),
    ('nagashi_1_7', None, '全体', 85.7),
    ('box_5', None, '全体', None),
]

print()
print(f"{'戦略':<35} | {'条件':<15} | {'2024ROI':>8} | {'2025ROI':>8} | {'差':>8} | {'レース':>6} | {'的中率':>6}")
print('-' * 100)

results = []
for formation, cond_func, cond_name, past_roi in tests:
    result = run_backtest(formation, cond_func, cond_name)
    
    past_roi_str = f"{past_roi:.1f}%" if past_roi else "N/A"
    diff = result['roi'] - past_roi if past_roi else 0
    diff_str = f"{diff:+.1f}%" if past_roi else "N/A"
    
    form_name = formation.replace('nagashi_1_', '1頭軸→相手Top').replace('nagashi_2_', '2頭軸→相手Top').replace('box_', 'Box Top')
    
    print(f"{form_name:<35} | {cond_name:<15} | {past_roi_str:>8} | {result['roi']:>7.1f}% | {diff_str:>8} | {result['races']:>6} | {result['hit_rate']:>5.1f}%")
    
    results.append({
        'formation': formation,
        'condition': cond_name,
        'past_roi': past_roi,
        'current_roi': result['roi'],
        'races': result['races'],
        'hit_rate': result['hit_rate']
    })

# ROI 100%超の戦略
print()
print('=' * 70)
print('🏆 2025年v7でROI 100%以上の戦略:')
print('=' * 70)

over_100 = [r for r in results if r['current_roi'] >= 100]
if over_100:
    for r in sorted(over_100, key=lambda x: x['current_roi'], reverse=True):
        print(f"  {r['formation']} x {r['condition']}: ROI {r['current_roi']:.1f}% ({r['races']}レース)")
else:
    print("  ⚠️ ROI 100%超の戦略は見つかりませんでした")

# 改善した戦略
print()
print('📈 2024年より改善した戦略:')
improved = [r for r in results if r['past_roi'] and r['current_roi'] > r['past_roi']]
for r in sorted(improved, key=lambda x: x['current_roi'] - x['past_roi'], reverse=True)[:5]:
    diff = r['current_roi'] - r['past_roi']
    print(f"  {r['formation']} x {r['condition']}: {r['past_roi']:.1f}% → {r['current_roi']:.1f}% (+{diff:.1f}%)")

print()
print('=' * 70)

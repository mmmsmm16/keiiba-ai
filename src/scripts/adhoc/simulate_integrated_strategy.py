import pandas as pd
from collections import defaultdict
from itertools import combinations, permutations
import sys

# Load v7 predictions (2025)
v7_path = 'experiments/v7_ensemble_full/reports/predictions.parquet'
try:
    df = pd.read_parquet(v7_path)
except FileNotFoundError:
    print(f"File not found: {v7_path}")
    sys.exit(1)

df['date'] = pd.to_datetime(df['date'])
df = df[df['date'].dt.year == 2025]
df['pred_rank'] = df.groupby('race_id')['score'].rank(method='first', ascending=False)

# Load payouts
pay_path = 'experiments/payouts_2024_2025.parquet'
try:
    pay_df = pd.read_parquet(pay_path)
except FileNotFoundError:
    print(f"File not found: {pay_path}")
    sys.exit(1)

payout_map = defaultdict(lambda: {'sanrentan': {}})
for _, row in pay_df.iterrows():
    rid = str(row['race_id'])
    for i in range(1, 7):
        col_a = f'haraimodoshi_sanrentan_{i}a'
        col_b = f'haraimodoshi_sanrentan_{i}b'
        if col_a in row and row[col_a]:
            try:
                payout_map[rid]['sanrentan'][str(row[col_a]).strip()] = int(float(str(row[col_b]).strip()))
            except: pass
payout_map = dict(payout_map)

# Build Race Data
race_data = {}
for rid, grp in df.groupby('race_id'):
    rid_str = str(rid)
    if rid_str not in payout_map: continue
    sorted_g = grp.sort_values('pred_rank')
    if len(sorted_g) < 6: continue
    
    horses = sorted_g['horse_number'].astype(int).tolist()
    pops = [int(sorted_g.iloc[i].get('popularity', 99)) if pd.notna(sorted_g.iloc[i].get('popularity')) else 99 for i in range(6)]
    scores = sorted_g['score'].tolist()
    
    race_data[rid_str] = {
        'horses': horses,
        'pops': pops,
        'scores': scores,
        'gap_1_2': scores[0] - scores[1] if len(scores) > 1 else 0
    }

# Split Data
df_q1 = df[df['date'].dt.month <= 3]
df_q2 = df[df['date'].dt.month >= 4]

print(f'Total Races: {len(race_data)} (Q1: {len([r for r in race_data if df[df.race_id==r].date.iloc[0].month<=3])}, Q2+: {len([r for r in race_data if df[df.race_id==r].date.iloc[0].month>=4])})')

def run_sim(target_rids, label):
    stats = {
        'A': {'cost': 0, 'ret': 0, 'races': 0}, 
        'B': {'cost': 0, 'ret': 0, 'races': 0}, 
        'Total': {'cost': 0, 'ret': 0, 'races': 0} 
    }

    for rid in target_rids:
        if rid not in race_data: continue
        rd = race_data[rid]
        horses = rd['horses']
        pop1 = rd['pops'][0]
        gap = rd['gap_1_2']
        
        bet_pattern = None
        axis_idx = 0
        
        if pop1 >= 4:
            bet_pattern = 'A'
            axis_idx = 0 # 1st
        elif gap < 0.05:
            bet_pattern = 'B'
            axis_idx = 1 # 2nd
        
        if bet_pattern:
            axis = horses[axis_idx]
            opps = [h for i, h in enumerate(horses[:5]) if i != axis_idx]
            
            cost = 0
            ret = 0
            for o1, o2 in permutations(opps, 2):
                cost += 100
                key = str(axis).zfill(2) + str(o1).zfill(2) + str(o2).zfill(2)
                if key in payout_map[rid]['sanrentan']:
                    ret += payout_map[rid]['sanrentan'][key]
            
            stats[bet_pattern]['cost'] += cost
            stats[bet_pattern]['ret'] += ret
            stats[bet_pattern]['races'] += 1
            stats['Total']['cost'] += cost
            stats['Total']['ret'] += ret
            stats['Total']['races'] += 1

    print(f'\n=== Integrated Strategy 2025 ({label}) ===')
    print(f'{"Pattern":<30} | {"Races":>6} | {"Cost":>10} | {"Return":>10} | {"Profit":>10} | {"ROI":>6}')
    print('-' * 90)

    for pat in ['A', 'B', 'Total']:
        s = stats[pat]
        roi = s['ret'] / s['cost'] * 100 if s['cost'] > 0 else 0
        prof = s['ret'] - s['cost']
        name = "Pop>=4 (Axis 1st)" if pat == 'A' else "Pop<4 & Gap<0.05 (Axis 2nd)" if pat == 'B' else "TOTAL"
        print(f'{name:<30} | {s["races"]:>6} | {s["cost"]:>10,} | {s["ret"]:>10,} | {prof:>10,} | {roi:>5.1f}%')

# Run for Q1
q1_rids = df_q1['race_id'].astype(str).tolist()
run_sim(q1_rids, "Q1: Jan-Mar")

# Run for Q2+
q2_rids = df_q2['race_id'].astype(str).tolist()
run_sim(q2_rids, "Q2+: Apr-Dec")

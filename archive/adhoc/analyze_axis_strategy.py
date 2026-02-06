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
    # pop取得 (rank1, rank2, rank3)
    pops = [int(sorted_g.iloc[i].get('popularity', 99)) if pd.notna(sorted_g.iloc[i].get('popularity')) else 99 for i in range(6)]
    scores = sorted_g['score'].tolist()
    
    # Target: Races where Top1 prediction is rank 1-3 popularity
    if pops[0] < 4:
        # Calculate gaps
        gap_1_2 = scores[0] - scores[1] if len(scores) > 1 else 0
        gap_2_3 = scores[1] - scores[2] if len(scores) > 2 else 0
        
        race_data[rid_str] = {
            'horses': horses,
            'pops': pops,
            'scores': scores,
            'gap_1_2': gap_1_2,
            'gap_2_3': gap_2_3,
            'axis_2_pop': pops[1], # 2nd predicted horse popularity
            'axis_3_pop': pops[2], # 3rd predicted horse popularity
        }

print(f'Target Races (Top1 is Pop<4): {len(race_data)}')

def calc_roi(rids, axis_idx, payout_map, race_data):
    if not rids: return 0, 0, 0, 0
    cost = ret = races = 0
    for rid in rids:
        rd = race_data[rid]
        horses = rd['horses']
        axis = horses[axis_idx]
        
        # 1-axis 4-horse flow (opponents: rank 1-5 excluding axis)
        # Note: opps should be taken from pred_rank order.
        # Original logic: axis is horses[axis_idx]. 
        # Opponents are horses[:5] excluding axis.
        opps = [h for i, h in enumerate(horses[:5]) if i != axis_idx]
        
        for o1, o2 in permutations(opps, 2):
            cost += 100
            key = str(axis).zfill(2) + str(o1).zfill(2) + str(o2).zfill(2)
            if key in payout_map[rid]['sanrentan']:
                ret += payout_map[rid]['sanrentan'][key]
        races += 1
    return races, cost, ret, ret/cost*100 if cost>0 else 0

# Grid Search for Axis=2nd Strategy
print('\n=== Strategy: Axis = 2nd Predicted Horse ===')
print('(Results for races where Top1 is Pop<4)')
print(f'{"Condition":<30} | {"Races":>6} | {"Rate":>6} | {"Profit":>10} | {"ROI":>6}')
print('-' * 80)

filters = [
    ('All', lambda r: True),
    ('Axis2 Pop >= 3', lambda r: r['axis_2_pop'] >= 3),
    ('Axis2 Pop >= 4', lambda r: r['axis_2_pop'] >= 4),
    ('Axis2 Pop >= 5', lambda r: r['axis_2_pop'] >= 5),
    ('Axis2 Pop >= 6', lambda r: r['axis_2_pop'] >= 6),
    ('Gap(1-2) < 0.1', lambda r: r['gap_1_2'] < 0.1),
    ('Gap(1-2) < 0.05', lambda r: r['gap_1_2'] < 0.05),
    ('Gap(1-2) >= 0.1', lambda r: r['gap_1_2'] >= 0.1),
    ('Gap(1-2) >= 0.2', lambda r: r['gap_1_2'] >= 0.2), 
    ('Ax2Pop>=4 & Gap<0.1', lambda r: r['axis_2_pop'] >= 4 and r['gap_1_2'] < 0.1),
    ('Ax2Pop>=5 & Gap<0.1', lambda r: r['axis_2_pop'] >= 5 and r['gap_1_2'] < 0.1),
]

results = []
for name, f in filters:
    rids = [rid for rid, rd in race_data.items() if f(rd)]
    races, cost, ret, roi = calc_roi(rids, 1, payout_map, race_data) # axis_idx=1 (2nd)
    rate = races / len(race_data) * 100 if len(race_data) > 0 else 0
    profit = ret - cost
    results.append((name, races, rate, profit, roi))

results.sort(key=lambda x: x[4], reverse=True)
for row in results:
    s_cond = row[0]
    s_races = str(row[1])
    s_rate = f'{row[2]:.1f}%'
    s_profit = f'{row[3]:,}'
    s_roi = f'{row[4]:.1f}%'
    print(f'{s_cond:<30} | {s_races:>6} | {s_rate:>6} | {s_profit:>10} | {s_roi:>6}')

# Grid Search for Axis=3rd Strategy
print('\n=== Strategy: Axis = 3rd Predicted Horse ===')
print(f'{"Condition":<30} | {"Races":>6} | {"Rate":>6} | {"Profit":>10} | {"ROI":>6}')
print('-' * 80)

filters_3rd = [
    ('All', lambda r: True),
    ('Axis3 Pop >= 5', lambda r: r['axis_3_pop'] >= 5),
    ('Axis3 Pop >= 7', lambda r: r['axis_3_pop'] >= 7),
    ('Gap(2-3) < 0.1', lambda r: r['gap_2_3'] < 0.1),
    ('Ax3Pop>=5 & Gap<0.1', lambda r: r['axis_3_pop'] >= 5 and r['gap_2_3'] < 0.1),
]

results_3 = []
for name, f in filters_3rd:
    rids = [rid for rid, rd in race_data.items() if f(rd)]
    races, cost, ret, roi = calc_roi(rids, 2, payout_map, race_data) # axis_idx=2 (3rd)
    rate = races / len(race_data) * 100 if len(race_data) > 0 else 0
    profit = ret - cost
    results_3.append((name, races, rate, profit, roi))

results_3.sort(key=lambda x: x[4], reverse=True)
for row in results_3:
    s_cond = row[0]
    s_races = str(row[1])
    s_rate = f'{row[2]:.1f}%'
    s_profit = f'{row[3]:,}'
    s_roi = f'{row[4]:.1f}%'
    print(f'{s_cond:<30} | {s_races:>6} | {s_rate:>6} | {s_profit:>10} | {s_roi:>6}')

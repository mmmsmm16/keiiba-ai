import pandas as pd
from collections import defaultdict
from itertools import combinations, permutations
import sys
import numpy as np

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

# Q1/Q2 Split
df_q1 = df[df['date'].dt.month <= 3]
df_q2 = df[df['date'].dt.month >= 4]

# Load payouts
pay_path = 'experiments/payouts_2024_2025.parquet'
try:
    pay_df = pd.read_parquet(pay_path)
except FileNotFoundError:
    print(f"File not found: {pay_path}")
    sys.exit(1)

payout_map = defaultdict(lambda: {'tansho': {}, 'umaren': {}, 'wide': {}, 'sanrenpuku': {}, 'sanrentan': {}})
for _, row in pay_df.iterrows():
    rid = str(row['race_id'])
    for prefix in ['haraimodoshi_tansho', 'haraimodoshi_umaren', 'haraimodoshi_wide', 'haraimodoshi_sanrenpuku', 'haraimodoshi_sanrentan']:
        bet_type = prefix.split('_')[1]
        max_count = 3 if bet_type != 'sanrentan' else 6
        for i in range(1, max_count + 1):
            col_a = prefix + '_' + str(i) + 'a'
            col_b = prefix + '_' + str(i) + 'b'
            if col_a in row and row[col_a]:
                try:
                    payout_map[rid][bet_type][str(row[col_a]).strip()] = int(float(str(row[col_b]).strip()))
                except: pass
payout_map = dict(payout_map)

# Precompute Race Data
def build_race_data(target_df):
    rd = {}
    for rid, grp in target_df.groupby('race_id'):
        rid_str = str(rid)
        if rid_str not in payout_map: continue
        sorted_g = grp.sort_values('pred_rank')
        if len(sorted_g) < 6: continue
        
        horses = sorted_g['horse_number'].astype(int).tolist()
        pops = [int(sorted_g.iloc[i].get('popularity', 99)) if pd.notna(sorted_g.iloc[i].get('popularity')) else 99 for i in range(6)]
        scores = sorted_g['score'].tolist()
        
        # Only interested in races where 1st prediction is Favorite (Pop < 4)
        if pops[0] >= 4: continue
            
        rd[rid_str] = {
            'horses': horses,
            'pops': pops,
            'gap_1_2': scores[0] - scores[1] if len(scores) > 1 else 0,
            'distance': grp['distance'].iloc[0],
            'surface': str(grp['surface'].iloc[0]), # 1:芝, 2:ダート
            'venue': rid_str[4:6],
            'weather': str(grp['weather'].iloc[0]),
            'state': str(grp['state'].iloc[0]), # 1:良 2:稍重 3:重 4:不良
            # Assuming title or other keywords might indicate Handicap, but simpler to use available cols
            'is_sprint': grp['distance'].iloc[0] <= 1400,
            'is_long': grp['distance'].iloc[0] >= 2200,
            'is_bad_state': int(grp['state'].iloc[0]) >= 3 if pd.notna(grp['state'].iloc[0]) and str(grp['state'].iloc[0]).isdigit() else False
        }
    return rd

q1_rd = build_race_data(df_q1)
q2_rd = build_race_data(df_q2)

print(f'Races (Pop<4): Q1={len(q1_rd)}, Q2+={len(q2_rd)}')

def calc_roi(rids, payout_map, race_data, bet_type='sanrentan_1axis_4opps'):
    if not rids: return 0, 0, 0
    cost = ret = 0
    for rid in rids:
        rd = race_data[rid]
        horses = rd['horses']
        axis = horses[0] # Always 1st predicted horse as axis for now
        
        if bet_type == 'sanrentan_1axis_4opps':
            opps = horses[1:5]
            for o1, o2 in permutations(opps, 2):
                cost += 100
                key = str(axis).zfill(2) + str(o1).zfill(2) + str(o2).zfill(2)
                if key in payout_map[rid]['sanrentan']: ret += payout_map[rid]['sanrentan'][key]
                
        elif bet_type == 'sanrenpuku_box_5':
            # Box 5 horses (1-5)
            box = horses[:5]
            for t in combinations(box, 3):
                cost += 100
                c_sorted = sorted(t)
                key = str(c_sorted[0]).zfill(2) + str(c_sorted[1]).zfill(2) + str(c_sorted[2]).zfill(2)
                if key in payout_map[rid]['sanrenpuku']: ret += payout_map[rid]['sanrenpuku'][key]

        elif bet_type == 'umaren_box_5':
            box = horses[:5]
            for t in combinations(box, 2):
                cost += 100
                c_sorted = sorted(t)
                key = str(c_sorted[0]).zfill(2) + str(c_sorted[1]).zfill(2)
                if key in payout_map[rid]['umaren']: ret += payout_map[rid]['umaren'][key]
                
        elif bet_type == 'wide_box_5':
            box = horses[:5]
            for t in combinations(box, 2):
                cost += 100
                c_sorted = sorted(t)
                key = str(c_sorted[0]).zfill(2) + str(c_sorted[1]).zfill(2)
                if key in payout_map[rid]['wide']: ret += payout_map[rid]['wide'][key]
        
        elif bet_type == 'sanrenpuku_1axis_5opps':
            opps = horses[1:6]
            for t in combinations([axis] + opps, 3):
                if axis in t:
                    cost += 100
                    c_sorted = sorted(t)
                    key = str(c_sorted[0]).zfill(2) + str(c_sorted[1]).zfill(2) + str(c_sorted[2]).zfill(2)
                    if key in payout_map[rid]['sanrenpuku']: ret += payout_map[rid]['sanrenpuku'][key]

    return cost, ret, ret/cost*100 if cost>0 else 0

# Conditions to test
conditions = [
    ('All', lambda r: True),
    ('Sprint (<=1400m)', lambda r: r['is_sprint']),
    ('Long (>=2200m)', lambda r: r['is_long']),
    ('Bad State (>=3)', lambda r: r['is_bad_state']),
    ('Gap < 0.05', lambda r: r['gap_1_2'] < 0.05),
    ('Gap < 0.1', lambda r: r['gap_1_2'] < 0.1),
    ('Gap >= 0.1', lambda r: r['gap_1_2'] >= 0.1),
    ('Dirt', lambda r: str(r['surface']) == '2'),
    ('Turf', lambda r: str(r['surface']) == '1'),
    ('Venue: Nakayama (06)', lambda r: r['venue'] == '06'),
    ('Venue: Hanshin (09)', lambda r: r['venue'] == '09'),
    ('Venue: Tokyo (05)', lambda r: r['venue'] == '05'),
    ('Venue: Kyoto (08)', lambda r: r['venue'] == '08'),
]

# Bet types to test
bet_types = [
    'sanrentan_1axis_4opps', # Standard
    'sanrenpuku_1axis_5opps', # Option C default
    'sanrenpuku_box_5', # Box
    'umaren_box_5',
    'wide_box_5'
]

print('\n=== Grid Search: Top1=Favorite(Pop<4) Races ===')
print(f'{"Condition":<25} | {"Bet Type":<22} | {"Q1 ROI":>7} | {"Q2+ ROI":>7} | {"Q2+ Races":>6}')
print('-' * 90)

results = []
for cond_name, cond_func in conditions:
    q1_target = [rid for rid, rd in q1_rd.items() if cond_func(rd)]
    q2_target = [rid for rid, rd in q2_rd.items() if cond_func(rd)]
    
    for bt in bet_types:
        _, _, q1_roi = calc_roi(q1_target, payout_map, q1_rd, bt)
        _, _, q2_roi = calc_roi(q2_target, payout_map, q2_rd, bt)
        
        results.append({
            'Condition': cond_name,
            'BetType': bt,
            'Q1_ROI': q1_roi,
            'Q2_ROI': q2_roi,
            'Q2_Races': len(q2_target)
        })

# Sort by Q1 ROI (Simulation mindset)
results.sort(key=lambda x: x['Q1_ROI'], reverse=True)

# Show Top 20
for r in results[:20]:
    print(f'{r["Condition"]:<25} | {r["BetType"]:<22} | {r["Q1_ROI"]:>6.1f}% | {r["Q2_ROI"]:>6.1f}% | {r["Q2_Races"]:>6}')

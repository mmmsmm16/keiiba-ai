"""
Box / Nagashi / Formation Betting Simulation
=============================================
Tests various multi-horse betting strategies for umaren/wide.
"""
import pandas as pd
import numpy as np
import joblib
from itertools import combinations
from sqlalchemy import create_engine, text

# Database connection
engine = create_engine('postgresql://postgres:postgres@db:5432/pckeiba')

# Load payout data
print("Loading payout data...")
payout_query = """
SELECT 
    CONCAT(kaisai_nen, keibajo_code, LPAD(kaisai_kai::text, 2, '0'), 
           LPAD(kaisai_nichime::text, 2, '0'), LPAD(race_bango::text, 2, '0')) as race_id,
    haraimodoshi_umaren_1a as umaren_pair,
    haraimodoshi_umaren_1b as umaren_payout,
    haraimodoshi_wide_1a as wide_pair_1, haraimodoshi_wide_1b as wide_payout_1,
    haraimodoshi_wide_2a as wide_pair_2, haraimodoshi_wide_2b as wide_payout_2,
    haraimodoshi_wide_3a as wide_pair_3, haraimodoshi_wide_3b as wide_payout_3
FROM jvd_hr
WHERE kaisai_nen >= '2024'
"""
payouts = pd.read_sql(text(payout_query), engine)

def parse_pair(pair_str):
    if pd.isna(pair_str) or len(str(pair_str)) < 4:
        return (0, 0)
    try:
        s = str(pair_str).zfill(4)
        h1, h2 = int(s[:2]), int(s[2:4])
        return (min(h1, h2), max(h1, h2))
    except:
        return (0, 0)

def parse_payout(payout_str):
    try:
        return int(payout_str) / 100
    except:
        return 0

payouts['umaren_horses'] = payouts['umaren_pair'].apply(parse_pair)
payouts['umaren_odds'] = payouts['umaren_payout'].apply(parse_payout)
for i in [1, 2, 3]:
    payouts[f'wide_horses_{i}'] = payouts[f'wide_pair_{i}'].apply(parse_pair)
    payouts[f'wide_odds_{i}'] = payouts[f'wide_payout_{i}'].apply(parse_payout)

# Load prediction data
print("Loading prediction data...")
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df = df[(df['year'] == 2024) & (df['odds'] > 0)].copy()

model = joblib.load('models/experiments/exp_lambdarank_v12_batch4_optuna/model.pkl')
features = pd.read_csv('models/experiments/exp_lambdarank_v12_batch4_optuna/features.csv')['0'].tolist()
y_pred = model.predict(df[features].values)
df['pred_score'] = y_pred

# Softmax probs
def softmax_probs(group):
    scores = group['pred_score'].values
    exp_scores = np.exp(scores - np.max(scores))
    return pd.Series(exp_scores / exp_scores.sum(), index=group.index)

df['win_prob'] = df.groupby('race_id', group_keys=False).apply(softmax_probs)
df['ev'] = df['win_prob'] * df['odds']
df['pred_rank'] = df.groupby('race_id')['pred_score'].rank(ascending=False)
df['ev_rank'] = df.groupby('race_id')['ev'].rank(ascending=False)

# Get top horses per race
def get_race_horses(race_df, n):
    horses = race_df.nsmallest(n, 'pred_rank')['horse_number'].astype(int).tolist()
    return horses

def get_high_ev_horses(race_df, n, exclude=None):
    if exclude:
        race_df = race_df[~race_df['horse_number'].isin(exclude)]
    horses = race_df.nsmallest(n, 'ev_rank')['horse_number'].astype(int).tolist()
    return horses

# Simulate function
def simulate_strategy(strategy_name, get_pairs_fn):
    results = []
    for race_id in df['race_id'].unique():
        race_df = df[df['race_id'] == race_id]
        if len(race_df) < 3:
            continue
        
        pairs = get_pairs_fn(race_df)
        if not pairs:
            continue
        
        payout_row = payouts[payouts['race_id'] == race_id]
        if len(payout_row) == 0:
            continue
        payout_row = payout_row.iloc[0]
        
        # Check umaren
        umaren_hit = payout_row['umaren_horses']
        umaren_return = 0
        for p in pairs:
            if p == umaren_hit:
                umaren_return = payout_row['umaren_odds']
                break
        
        # Check wide (multiple hits possible)
        wide_return = 0
        for p in pairs:
            for i in [1, 2, 3]:
                if p == payout_row[f'wide_horses_{i}']:
                    wide_return += payout_row[f'wide_odds_{i}']
        
        results.append({
            'race_id': race_id,
            'num_pairs': len(pairs),
            'umaren_return': umaren_return,
            'wide_return': wide_return
        })
    
    res_df = pd.DataFrame(results)
    total_bets = res_df['num_pairs'].sum()
    total_umaren_bets = total_bets * 100
    total_wide_bets = total_bets * 100
    umaren_returns = res_df['umaren_return'].sum() * 100
    wide_returns = res_df['wide_return'].sum() * 100
    
    umaren_hits = (res_df['umaren_return'] > 0).sum()
    wide_hits = (res_df['wide_return'] > 0).sum()
    
    print(f"\n=== {strategy_name} ===")
    print(f"Races: {len(res_df)}, Total Pairs Bet: {total_bets}")
    print(f"馬連: Hits={umaren_hits}, Return={umaren_returns:,.0f}, ROI={umaren_returns/total_umaren_bets*100:.1f}%")
    print(f"ワイド: Hits={wide_hits}, Return={wide_returns:,.0f}, ROI={wide_returns/total_wide_bets*100:.1f}%")
    
    return res_df

# Strategy 1: BOX Top 3
print("\n" + "="*60)
def box_top3(race_df):
    horses = get_race_horses(race_df, 3)
    if len(horses) < 2:
        return []
    return [tuple(sorted(p)) for p in combinations(horses, 2)]

simulate_strategy("BOX Top 3 (3 pairs)", box_top3)

# Strategy 2: BOX Top 4
def box_top4(race_df):
    horses = get_race_horses(race_df, 4)
    if len(horses) < 2:
        return []
    return [tuple(sorted(p)) for p in combinations(horses, 2)]

simulate_strategy("BOX Top 4 (6 pairs)", box_top4)

# Strategy 3: Nagashi - Top 1 anchor + Top 2-5 partners
def nagashi_top1_to_5(race_df):
    anchor = get_race_horses(race_df, 1)
    if not anchor:
        return []
    partners = get_race_horses(race_df, 5)
    partners = [h for h in partners if h != anchor[0]][:4]
    return [tuple(sorted([anchor[0], p])) for p in partners]

simulate_strategy("流し: Top1軸 → Top2-5 (4 pairs)", nagashi_top1_to_5)

# Strategy 4: Formation - Top 1 (high prob) + Top 3 EV horses
def formation_prob_ev(race_df):
    anchor = get_race_horses(race_df, 1)
    if not anchor:
        return []
    high_ev = get_high_ev_horses(race_df, 3, exclude=anchor)
    return [tuple(sorted([anchor[0], h])) for h in high_ev]

simulate_strategy("フォーメーション: Top1(確率) + Top3 EV (3 pairs)", formation_prob_ev)

# Strategy 5: Formation - Top 2 (high prob) + Top 2 EV
def formation_2x2(race_df):
    anchors = get_race_horses(race_df, 2)
    high_ev = get_high_ev_horses(race_df, 2, exclude=anchors)
    pairs = []
    for a in anchors:
        for e in high_ev:
            pairs.append(tuple(sorted([a, e])))
    # Also add anchors pair
    if len(anchors) == 2:
        pairs.append(tuple(sorted(anchors)))
    return pairs

simulate_strategy("フォーメーション: Top2確率 x Top2 EV + 確率同士 (5 pairs)", formation_2x2)

print("\n" + "="*60)
print("DONE")

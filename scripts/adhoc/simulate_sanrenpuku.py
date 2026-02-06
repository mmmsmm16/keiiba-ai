"""
Sanrenpuku (3renpuku) Betting Simulation
=========================================
Tests Top 5 BOX and Nagashi strategies for Sanrenpuku.
"""
import pandas as pd
import numpy as np
import joblib
from itertools import combinations
from sqlalchemy import create_engine, text

# Database connection
engine = create_engine('postgresql://postgres:postgres@db:5432/pckeiba')

# Load payout data (Sanrenpuku)
print("Loading payout data...")
payout_query = """
SELECT 
    CONCAT(kaisai_nen, keibajo_code, LPAD(kaisai_kai::text, 2, '0'), 
           LPAD(kaisai_nichime::text, 2, '0'), LPAD(race_bango::text, 2, '0')) as race_id,
    haraimodoshi_sanrenpuku_1a as sanrenpuku_trio_1,
    haraimodoshi_sanrenpuku_1b as sanrenpuku_payout_1,
    haraimodoshi_sanrenpuku_2a as sanrenpuku_trio_2, 
    haraimodoshi_sanrenpuku_2b as sanrenpuku_payout_2,
    haraimodoshi_sanrenpuku_3a as sanrenpuku_trio_3, 
    haraimodoshi_sanrenpuku_3b as sanrenpuku_payout_3
FROM jvd_hr
WHERE kaisai_nen >= '2024'
"""
payouts = pd.read_sql(text(payout_query), engine)

def parse_trio(trio_str):
    if pd.isna(trio_str) or len(str(trio_str)) < 6:
        return (0, 0, 0)
    try:
        s = str(trio_str).zfill(6)
        h1, h2, h3 = int(s[:2]), int(s[2:4]), int(s[4:6])
        return tuple(sorted([h1, h2, h3]))
    except:
        return (0, 0, 0)

def parse_payout(payout_str):
    try:
        return int(payout_str) / 100
    except:
        return 0

for i in [1, 2, 3]:
    payouts[f'trio_{i}'] = payouts[f'sanrenpuku_trio_{i}'].apply(parse_trio)
    payouts[f'odds_{i}'] = payouts[f'sanrenpuku_payout_{i}'].apply(parse_payout)

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
df['pred_rank'] = df.groupby('race_id')['pred_score'].rank(ascending=False)
df = df.sort_values(['race_id', 'pred_rank'])

def get_race_horses(race_df, n):
    return race_df.nsmallest(n, 'pred_rank')['horse_number'].astype(int).tolist()

def simulate(strategy_name, get_bets_fn):
    results = []
    race_groups = df.groupby('race_id')
    
    for race_id, race_df in race_groups:
        bets = get_bets_fn(race_df)
        if not bets: continue
        
        payout_row = payouts[payouts['race_id'] == race_id]
        if len(payout_row) == 0: continue
        payout_row = payout_row.iloc[0]
        
        ret = 0
        hits = 0
        
        for b in bets:
            for i in [1, 2, 3]: # Check up to 3 payouts (doshaku)
                if b == payout_row[f'trio_{i}']:
                    ret += payout_row[f'odds_{i}']
                    hits = 1
                    
        results.append({
            'bets': len(bets),
            'return': ret,
            'hit': hits
        })
        
    res_df = pd.DataFrame(results)
    if len(res_df) == 0: return

    total_bet = res_df['bets'].sum() * 100
    total_ret = res_df['return'].sum() * 100
    total_hits = res_df['hit'].sum()
    roi = total_ret / total_bet * 100
    
    print(f"=== {strategy_name} ===")
    print(f"Races: {len(res_df)}, Total Bets: {int(total_bet/100)}")
    print(f"Hits: {total_hits}, Hit Rate: {total_hits/len(res_df)*100:.1f}%")
    print(f"Return: {total_ret:,.0f}, ROI: {roi:.1f}%")
    print()

print("\nRunning Sanrenpuku Simulations...")

# 1. Top 5 BOX (10 bets)
def box_top5(race_df):
    horses = get_race_horses(race_df, 5)
    if len(horses) < 3: return []
    return [tuple(sorted(p)) for p in combinations(horses, 3)]

simulate("Top 5 BOX (10点)", box_top5)

# 2. Top 1 Nagashi (Axis) -> Partner Top 2-6 (10 bets)
def nagashi_top1_partners5(race_df):
    horses = get_race_horses(race_df, 6) # Need rank 6 to get 5 partners excluding rank 1
    anchor = horses[0]
    partners = horses[1:6]
    return [tuple(sorted([anchor] + list(p))) for p in combinations(partners, 2)]

simulate("Top 1 軸1頭ながし (相手5頭=10点)", nagashi_top1_partners5)

# 3. Top 1-2 Axis -> Partner Top 3-7 (5 bets)
def nagashi_top2_partners5(race_df):
    horses = get_race_horses(race_df, 7)
    anchors = horses[:2] # Rank 1, 2
    partners = horses[2:7] # Rank 3, 4, 5, 6, 7
    return [tuple(sorted(anchors + [p])) for p in partners]

simulate("Top 1-2 軸2頭ながし (相手5頭=5点)", nagashi_top2_partners5)

# 4. Top 5 BOX + Odd Filter (Skip if Top1 Odds < 3.0)
def box_top5_odds_filter(race_df):
    if race_df.iloc[0]['odds'] < 3.0: return []
    horses = get_race_horses(race_df, 5)
    if len(horses) < 3: return []
    return [tuple(sorted(p)) for p in combinations(horses, 3)]

simulate("Top 5 BOX (Top1 Odds < 3.0 除外)", box_top5_odds_filter)

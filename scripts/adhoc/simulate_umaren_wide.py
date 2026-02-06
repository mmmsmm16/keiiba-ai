"""
Umaren / Wide Betting Simulation
================================
Simulates betting on model's top 2 predictions as umaren/wide.
"""
import pandas as pd
import numpy as np
import joblib
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

# Parse umaren pair (format: "0305" -> horses 3 and 5)
def parse_pair(pair_str):
    if pd.isna(pair_str) or len(pair_str) < 4:
        return (0, 0)
    try:
        h1 = int(pair_str[:2])
        h2 = int(pair_str[2:4])
        return (min(h1, h2), max(h1, h2))
    except:
        return (0, 0)

def parse_payout(payout_str):
    if pd.isna(payout_str):
        return 0
    try:
        return int(payout_str) / 100  # Convert to 100yen units
    except:
        return 0

payouts['umaren_horses'] = payouts['umaren_pair'].apply(parse_pair)
payouts['umaren_odds'] = payouts['umaren_payout'].apply(parse_payout)

# Parse wide (multiple hit possibilities)
for i in [1, 2, 3]:
    payouts[f'wide_horses_{i}'] = payouts[f'wide_pair_{i}'].apply(parse_pair)
    payouts[f'wide_odds_{i}'] = payouts[f'wide_payout_{i}'].apply(parse_payout)

print(f"Loaded {len(payouts)} race payouts")

# Load prediction data
print("Loading prediction data...")
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df = df[(df['year'] == 2024) & (df['odds'] > 0)].copy()
df['is_win'] = (df['rank'] == 1).astype(int)
df['is_top2'] = (df['rank'] <= 2).astype(int)

model = joblib.load('models/experiments/exp_lambdarank_v12_batch4_optuna/model.pkl')
features = pd.read_csv('models/experiments/exp_lambdarank_v12_batch4_optuna/features.csv')['0'].tolist()
y_pred = model.predict(df[features].values)
df['pred_score'] = y_pred
df['pred_rank'] = df.groupby('race_id')['pred_score'].rank(ascending=False)

# Get top 2 horses per race
top2 = df[df['pred_rank'] <= 2][['race_id', 'horse_number', 'pred_rank', 'rank']].copy()
top2_pivot = top2.pivot(index='race_id', columns='pred_rank', values='horse_number').reset_index()
top2_pivot.columns = ['race_id', 'pred_1', 'pred_2']
top2_pivot['pred_pair'] = top2_pivot.apply(
    lambda r: (int(min(r['pred_1'], r['pred_2'])), int(max(r['pred_1'], r['pred_2']))) 
    if pd.notna(r['pred_1']) and pd.notna(r['pred_2']) else (0, 0), axis=1
)

# Merge with payouts
print("Merging data...")
merged = top2_pivot.merge(payouts, on='race_id', how='inner')
print(f"Races with payouts: {len(merged)}")

# Check umaren hits
merged['umaren_hit'] = merged.apply(lambda r: r['pred_pair'] == r['umaren_horses'], axis=1)

# Check wide hits (any of the 3 wide combinations)
def check_wide_hit(row):
    pred = row['pred_pair']
    for i in [1, 2, 3]:
        if pred == row[f'wide_horses_{i}']:
            return row[f'wide_odds_{i}']
    return 0

merged['wide_return'] = merged.apply(check_wide_hit, axis=1)
merged['wide_hit'] = merged['wide_return'] > 0

# Results
print()
print("=== UMAREN (Top1-Top2 as pair) ===")
total_bets = len(merged)
umaren_hits = merged['umaren_hit'].sum()
umaren_returns = merged[merged['umaren_hit']]['umaren_odds'].sum() * 100
umaren_roi = umaren_returns / (total_bets * 100) * 100
print(f"Bets: {total_bets}, Hits: {umaren_hits}, Hit Rate: {umaren_hits/total_bets*100:.1f}%")
print(f"Total Bet: {total_bets*100:,}, Return: {umaren_returns:,.0f}, ROI: {umaren_roi:.1f}%")

print()
print("=== WIDE (Top1-Top2 as pair) ===")
wide_hits = merged['wide_hit'].sum()
wide_returns = merged['wide_return'].sum() * 100
wide_roi = wide_returns / (total_bets * 100) * 100
print(f"Bets: {total_bets}, Hits: {wide_hits}, Hit Rate: {wide_hits/total_bets*100:.1f}%")
print(f"Total Bet: {total_bets*100:,}, Return: {wide_returns:,.0f}, ROI: {wide_roi:.1f}%")

# With EV filtering (using sum of top2 probabilities)
print()
print("=== With EV filtering ===")
# Get top2 probabilities
top2_probs = df[df['pred_rank'] <= 2].groupby('race_id').agg({
    'pred_score': 'sum'
}).reset_index()
top2_probs.columns = ['race_id', 'pair_score']
merged = merged.merge(top2_probs, on='race_id', how='left')
merged['pair_score_pct'] = merged['pair_score'].rank(pct=True)

for threshold in [0.5, 0.7, 0.9]:
    filtered = merged[merged['pair_score_pct'] >= threshold]
    if len(filtered) < 50:
        continue
    
    # Umaren
    u_hits = filtered['umaren_hit'].sum()
    u_roi = filtered[filtered['umaren_hit']]['umaren_odds'].sum() * 100 / (len(filtered) * 100) * 100
    
    # Wide
    w_hits = filtered['wide_hit'].sum()
    w_roi = filtered['wide_return'].sum() * 100 / (len(filtered) * 100) * 100
    
    print(f"Top {int((1-threshold)*100)}% confidence: Bets={len(filtered)}, Umaren={u_hits} hits ROI={u_roi:.1f}%, Wide={w_hits} hits ROI={w_roi:.1f}%")

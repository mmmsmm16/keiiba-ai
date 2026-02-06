"""
Optimize Rank 1 Odds Filter
===========================
Optimizes Min/Max Odds threshold for Rank 1 bets.
Strategy: Bet on Rank 1 horse IF MinOdds <= Odds <= MaxOdds
Target: Place (Fukusho), Win (Tansho)
"""
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
from scipy.special import softmax

# --- Load Data & Model ---
print("Loading data...")
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df_test = df[df['date'].dt.year == 2024].copy()
df_test = df_test[(df_test['odds'] > 0) & (df_test['odds'].notna())].reset_index(drop=True)

# Features
df_test['yoso_juni_num'] = pd.to_numeric(df_test['yoso_juni'], errors='coerce').fillna(8)
df_test['popularity_num'] = pd.to_numeric(df_test['popularity'], errors='coerce').fillna(8)
df_test['popularity_vs_yoso'] = df_test['popularity_num'] - df_test['yoso_juni_num']
df_test['odds_rank'] = df_test.groupby('race_id')['odds'].rank(ascending=True)
df_test['odds_rank_vs_yoso'] = df_test['odds_rank'] - df_test['yoso_juni_num']
if 'relative_horse_elo_z' in df_test.columns:
    df_test['elo_rank'] = df_test.groupby('race_id')['relative_horse_elo_z'].rank(ascending=False)
    df_test['odds_rank_vs_elo'] = df_test['odds_rank'] - df_test['elo_rank']
else:
    df_test['odds_rank_vs_elo'] = 0
df_test['is_high_odds'] = (df_test['odds'] >= 10).astype(int)
df_test['is_mid_odds'] = ((df_test['odds'] >= 5) & (df_test['odds'] < 10)).astype(int)

# Load Model
model_dir = 'models/experiments/exp_lambdarank_hard_weighted'
model = joblib.load(f'{model_dir}/model.pkl')
features = pd.read_csv(f'{model_dir}/features.csv')['0'].tolist()

# Predict
scores = model.predict(df_test[features].values)
df_test['pred_score'] = scores
df_test['pred_rank'] = df_test.groupby('race_id')['pred_score'].transform(lambda x: x.rank(ascending=False))

# Filter Rank 1
rank1 = df_test[df_test['pred_rank'] == 1].copy()
rank1['race_id'] = rank1['race_id'].astype(str)

print(f"Total Rank 1 Candidates: {len(rank1)}")

# Load Payouts
engine = create_engine('postgresql://postgres:postgres@db:5432/pckeiba')
payout_query = """
SELECT 
    CONCAT(kaisai_nen, keibajo_code, LPAD(kaisai_kai::text, 2, '0'), 
           LPAD(kaisai_nichime::text, 2, '0'), LPAD(race_bango::text, 2, '0')) as race_id,
    haraimodoshi_tansho_1a as win_horse, haraimodoshi_tansho_1b as win_payout,
    haraimodoshi_fukusho_1a as place_h_1, haraimodoshi_fukusho_1b as place_p_1,
    haraimodoshi_fukusho_2a as place_h_2, haraimodoshi_fukusho_2b as place_p_2,
    haraimodoshi_fukusho_3a as place_h_3, haraimodoshi_fukusho_3b as place_p_3
FROM jvd_hr WHERE kaisai_nen = '2024'
"""
payouts = pd.read_sql(text(payout_query), engine)
payouts['race_id'] = payouts['race_id'].astype(str)

# Global Merge
merged = rank1.merge(payouts, on='race_id', how='inner')
print(f"Merged Data Size: {len(merged)}")

# Pre-calculate returns
merged['win_ret'] = 0.0
merged['place_ret'] = 0.0

# Win
win_hit = (merged['horse_number'] == pd.to_numeric(merged['win_horse'], errors='coerce'))
merged.loc[win_hit, 'win_ret'] = pd.to_numeric(merged.loc[win_hit, 'win_payout'], errors='coerce') / 100

# Place
h_num_pad = merged['horse_number'].astype(str).str.zfill(2)
p1_hit = h_num_pad == merged['place_h_1'].astype(str)
p2_hit = h_num_pad == merged['place_h_2'].astype(str)
p3_hit = h_num_pad == merged['place_h_3'].astype(str)

merged.loc[p1_hit, 'place_ret'] = pd.to_numeric(merged.loc[p1_hit, 'place_p_1'], errors='coerce') / 100
merged.loc[p2_hit, 'place_ret'] = pd.to_numeric(merged.loc[p2_hit, 'place_p_2'], errors='coerce') / 100
merged.loc[p3_hit, 'place_ret'] = pd.to_numeric(merged.loc[p3_hit, 'place_p_3'], errors='coerce') / 100

# Optimize
min_odds_steps = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 2.0, 2.5]
max_odds_steps = [2.0, 3.0, 5.0, 10.0, 20.0, 999.0]

print("\n=== Win Strategy (Rank 1) ===")
print("MinOdd  MaxOdd  Bets   ROI      Profit   HitRate")
print("-" * 50)
for min_o in min_odds_steps:
    for max_o in max_odds_steps:
        if min_o >= max_o: continue
        
        target = merged[(merged['odds'] >= min_o) & (merged['odds'] <= max_o)]
        cnt = len(target)
        if cnt < 50: continue
        
        ret = target['win_ret'].sum()
        roi = ret / cnt * 100
        profit = (ret - cnt) * 100
        hit = len(target[target['win_ret'] > 0]) / cnt * 100
        
        if roi > 85: # Filter for interesting results
            print(f"{min_o:<7} {max_o:<7} {cnt:<6} {roi:<8.1f}% {profit:<8.0f} {hit:.1f}%")

print("\n=== Place Strategy (Rank 1) ===")
print("MinOdd  MaxOdd  Bets   ROI      Profit   HitRate")
print("-" * 50)
for min_o in min_odds_steps:
    for max_o in max_odds_steps:
        if min_o >= max_o: continue
        
        target = merged[(merged['odds'] >= min_o) & (merged['odds'] <= max_o)]
        cnt = len(target)
        if cnt < 50: continue
        
        ret = target['place_ret'].sum()
        roi = ret / cnt * 100
        profit = (ret - cnt) * 100
        hit = len(target[target['place_ret'] > 0]) / cnt * 100
        
        if roi > 85:
            print(f"{min_o:<7} {max_o:<7} {cnt:<6} {roi:<8.1f}% {profit:<8.0f} {hit:.1f}%")

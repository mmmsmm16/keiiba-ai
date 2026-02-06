"""
Optimize Wide Nagashi Strategy (Rank 1 Axis)
============================================
Search for >100% ROI using the Deep Value Model.
Strategy:
- Axis: Rank 1 (Fixed)
- Partners: Filtered by Rank or EV
- Filters: Axis Odds, Partner Odds, Combined Odds
"""
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
from scipy.special import softmax
import itertools

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

# Load Model (Deep Value)
model_dir = 'models/experiments/exp_lambdarank_hard_weighted'
model = joblib.load(f'{model_dir}/model.pkl')
features = pd.read_csv(f'{model_dir}/features.csv')['0'].tolist()

# Predict
scores = model.predict(df_test[features].values)
df_test['pred_score'] = scores
df_test['pred_prob'] = df_test.groupby('race_id')['pred_score'].transform(lambda x: softmax(x.values))
df_test['pred_rank'] = df_test.groupby('race_id')['pred_score'].transform(lambda x: x.rank(ascending=False))
df_test['win_ev'] = df_test['pred_prob'] * df_test['odds']

# Force string race_id
df_test['race_id'] = df_test['race_id'].astype(str)

# --- Load Payouts ---
print("Loading payout data...")
engine = create_engine('postgresql://postgres:postgres@db:5432/pckeiba')
payout_query = """
SELECT 
    CONCAT(kaisai_nen, keibajo_code, LPAD(kaisai_kai::text, 2, '0'), 
           LPAD(kaisai_nichime::text, 2, '0'), LPAD(race_bango::text, 2, '0')) as race_id,
    haraimodoshi_wide_1a as wide_p_1, haraimodoshi_wide_1b as wide_o_1,
    haraimodoshi_wide_2a as wide_p_2, haraimodoshi_wide_2b as wide_o_2,
    haraimodoshi_wide_3a as wide_p_3, haraimodoshi_wide_3b as wide_o_3
FROM jvd_hr WHERE kaisai_nen = '2024'
"""
payouts = pd.read_sql(text(payout_query), engine)
payouts['race_id'] = payouts['race_id'].astype(str)

def parse_pair(s):
    try:
        s = str(s).zfill(4)
        return tuple(sorted([int(s[:2]), int(s[2:4])]))
    except: return (0, 0)

for i in [1, 2, 3]:
    payouts[f'wide_pair_{i}'] = payouts[f'wide_p_{i}'].apply(parse_pair)
    payouts[f'wide_ret_{i}'] = pd.to_numeric(payouts[f'wide_o_{i}'], errors='coerce') / 100

payout_dict = payouts.set_index('race_id').to_dict('index')

# --- Simulation Logic ---

def simulate(axis_filter_func, partner_filter_func):
    """
    axis_filter_func: row -> bool
    partner_filter_func: row -> bool
    """
    total_bet = 0
    total_ret = 0
    
    # Pre-select valid axis horses
    # Assume Axis is ALWAYS Rank 1 (from previous analysis this is best)
    # Applied axis_filter on top of Rank 1
    
    # Group by race for partners
    # Optimize: Vectorized is hard for wide combinations.
    # Group loop is safer.
    
    for rid, grp in df_test.groupby('race_id'):
        # 1. Provide Axis
        axis_candidates = grp[grp['pred_rank'] == 1]
        if len(axis_candidates) == 0: continue
        axis = axis_candidates.iloc[0]
        
        if not axis_filter_func(axis): continue
        
        # 2. Provide Partners
        # Exclude axis from partners
        potential_partners = grp[grp['horse_number'] != axis['horse_number']]
        partners = potential_partners[potential_partners.apply(partner_filter_func, axis=1)]
        
        if len(partners) == 0: continue
        
        # 3. Bet
        axis_num = int(axis['horse_number'])
        p_nums = partners['horse_number'].astype(int).tolist()
        
        bet_pairs = [tuple(sorted([axis_num, p])) for p in p_nums]
        
        # 4. Check Result
        pay = payout_dict.get(rid)
        if not pay: continue
        
        race_ret = 0
        for bp in bet_pairs:
             if bp == pay['wide_pair_1']: race_ret += pay['wide_ret_1']
             elif bp == pay['wide_pair_2']: race_ret += pay['wide_ret_2']
             elif bp == pay['wide_pair_3']: race_ret += pay['wide_ret_3']
        
        total_bet += len(bet_pairs)
        total_ret += race_ret
        
    return total_bet, total_ret

print("\n=== Wide Nagashi Optimization (Deep Value) ===")

# Conditions
axis_odds_ranges = [(1.0, 2.0), (1.5, 3.0), (2.0, 5.0), (3.0, 10.0)]
partner_strategies = [
    ("Partner Rank 2-3", lambda r: 2 <= r['pred_rank'] <= 3),
    ("Partner Rank 2-5", lambda r: 2 <= r['pred_rank'] <= 5),
    ("Partner EV >= 0.8", lambda r: r['win_ev'] >= 0.8),
    ("Partner EV >= 1.0", lambda r: r['win_ev'] >= 1.0),
    ("Partner Odds 10-50", lambda r: 10 <= r['odds'] <= 50),
    ("Partner Odds 20+", lambda r: r['odds'] >= 20),
]

print(f"{'Axis Odds':<15} {'Partner':<20} {'Bets':<8} {'ROI':<8} {'Profit':<8}")
print("-" * 65)

for min_o, max_o in axis_odds_ranges:
    
    print(f"DEBUG: Testing Axis {min_o}-{max_o}")
    def axis_cond(r):
        return min_o <= r['odds'] <= max_o
    
    for p_name, p_func in partner_strategies:
        cnt, ret = simulate(axis_cond, p_func)
        
        if cnt > 10:
            roi = ret / cnt * 100
            profit = (ret - cnt) * 100
            print(f"{min_o}-{max_o:<11} {p_name:<20} {cnt:<8} {roi:<8.1f}% {profit:<8.0f}")

print("\nDONE")

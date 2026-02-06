"""
Optimize EV Thresholds for Deep Value Model
===========================================
Performs grid search on EV thresholds for Hard-Weighted model.
"""
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
from scipy.special import softmax
import itertools

# --- 1. Load Data & Model ---
print("Loading data...")
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df_test = df[df['date'].dt.year == 2024].copy()
df_test = df_test[(df_test['odds'] > 0) & (df_test['odds'].notna())].reset_index(drop=True)

# Add Undervalued Features
print("Adding features...")
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
print(f"Loading model from {model_dir}...")
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

print("DEBUG Stats:")
print(f"Test size: {len(df_test)}")
print(f"Win EV: mean={df_test['win_ev'].mean():.4f}")

# Rank 1 Stats
rank1 = df_test[df_test['pred_rank'] == 1]
print("\n=== Rank 1 Horses Stats ===")
print(rank1[['odds', 'pred_prob', 'win_ev']].describe())

# --- 2. Load Payout Data ---
print("Loading payout data...")
engine = create_engine('postgresql://postgres:postgres@db:5432/pckeiba')
payout_query = """
SELECT 
    CONCAT(kaisai_nen, keibajo_code, LPAD(kaisai_kai::text, 2, '0'), 
           LPAD(kaisai_nichime::text, 2, '0'), LPAD(race_bango::text, 2, '0')) as race_id,
    haraimodoshi_tansho_1a as win_horse, haraimodoshi_tansho_1b as win_payout,
    haraimodoshi_fukusho_1a as place_h_1, haraimodoshi_fukusho_1b as place_p_1,
    haraimodoshi_fukusho_2a as place_h_2, haraimodoshi_fukusho_2b as place_p_2,
    haraimodoshi_fukusho_3a as place_h_3, haraimodoshi_fukusho_3b as place_p_3,
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

# --- 3. Grid Search Logic ---
conditions = [
    ("Win All", None),
    ("Place All", None),
    ("Wide Box Candidates", None)
]

ev_thresholds = [1.0, 1.5, 2.0, 2.3, 2.5, 3.0]
odds_caps = [10, 15, 20, 30, 50, 99.9]

print("\n=== Optimization Results ===")

for name, _ in conditions:
    print(f"\n--- {name} ---")
    print(f"{'EV >=':<8} {'Odds <=':<8} {'Bets':<6} {'ROI':<8} {'Profit':<8}")
    print("-" * 45)
    
    for ev_min in ev_thresholds:
        for odds_max in odds_caps:
            
            # Filtered DF
            candidates = df_test[(df_test['win_ev'] >= ev_min) & (df_test['odds'] <= odds_max)]
            if len(candidates) == 0: continue
            
            cnt = 0
            ret = 0
            
            if name == "Win All":
                bets = candidates
                merged = bets.merge(payouts[['race_id', 'win_horse', 'win_payout']], on='race_id', how='inner')
                if len(merged) == 0: continue

                merged['hit'] = (merged['horse_number'] == pd.to_numeric(merged['win_horse'], errors='coerce'))
                merged['return'] = np.where(merged['hit'], pd.to_numeric(merged['win_payout'], errors='coerce')/100, 0.0)
                
                cnt = len(merged)
                ret = merged['return'].sum()
                
            elif name == "Place All":
                bets = candidates
                merged = bets.merge(payouts, on='race_id', how='inner')
                if len(merged) == 0: continue
                
                h_num_pad = merged['horse_number'].astype(str).str.zfill(2)
                
                hit1 = h_num_pad == merged['place_h_1'].astype(str)
                hit2 = h_num_pad == merged['place_h_2'].astype(str)
                hit3 = h_num_pad == merged['place_h_3'].astype(str)
                
                merged['return'] = 0.0
                merged.loc[hit1, 'return'] = pd.to_numeric(merged.loc[hit1, 'place_p_1'], errors='coerce')/100
                merged.loc[hit2, 'return'] = pd.to_numeric(merged.loc[hit2, 'place_p_2'], errors='coerce')/100
                merged.loc[hit3, 'return'] = pd.to_numeric(merged.loc[hit3, 'place_p_3'], errors='coerce')/100
                
                cnt = len(merged)
                ret = merged['return'].sum()

            elif name == "Wide Box Candidates":
                cnt = 0
                ret = 0
                payout_dict = payouts.set_index('race_id').to_dict('index')
                
                for rid, grp in candidates.groupby('race_id'):
                    if len(grp) < 2: continue
                    
                    horses = grp['horse_number'].astype(int).tolist()
                    if len(horses) > 5:
                         horses = grp.nlargest(5, 'win_ev')['horse_number'].astype(int).tolist()
                    
                    pairs = list(itertools.combinations(horses, 2))
                    bet_pairs = [tuple(sorted(p)) for p in pairs]
                    
                    pay = payout_dict.get(rid)
                    if not pay: continue
                    
                    race_ret = 0
                    for bp in bet_pairs:
                        if bp == pay['wide_pair_1']: race_ret += pay['wide_ret_1']
                        elif bp == pay['wide_pair_2']: race_ret += pay['wide_ret_2']
                        elif bp == pay['wide_pair_3']: race_ret += pay['wide_ret_3']
                    
                    cnt += len(bet_pairs)
                    ret += race_ret
            
            # Print results
            if cnt > 50:
                roi = (ret / cnt) * 100
                profit = (ret - cnt) * 100
                if roi > 80: # Show loosely good results
                    print(f"{ev_min:<8.1f} {odds_max:<8} {cnt:<6} {roi:<8.1f}% {profit:<8.0f}")

print("\nDONE")

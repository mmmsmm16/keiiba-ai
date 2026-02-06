"""
Refined Betting Simulation (BOX/Nagashi + Filters)
==================================================
1. Top 3 BOX with Odds filter
2. Top 1 Nagashi with EV filter (partners must have EV > threshold)
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
df = df.sort_values(['race_id', 'pred_rank'])

# --- Simulation Logic ---

def run_simulation(strategy_name, bet_decision_fn):
    results = []
    
    # Pre-calculate race data to speed up
    race_groups = df.groupby('race_id')
    
    for race_id, race_df in race_groups:
        # race_df is already sorted by pred_rank
        
        # Determine valid pairs to bet
        pairs_to_bet = bet_decision_fn(race_df)
        if not pairs_to_bet:
            continue
            
        # Get payouts
        payout_row = payouts[payouts['race_id'] == race_id]
        if len(payout_row) == 0:
            continue
        payout_row = payout_row.iloc[0]
        
        umaren_ret = 0
        wide_ret = 0
        
        for p in pairs_to_bet:
            # Umaren check
            if p == payout_row['umaren_horses']:
                umaren_ret += payout_row['umaren_odds']
            
            # Wide check
            for i in [1, 2, 3]:
                if p == payout_row[f'wide_horses_{i}']:
                    wide_ret += payout_row[f'wide_odds_{i}']
        
        results.append({
            'num_pairs': len(pairs_to_bet),
            'umaren_return': umaren_ret,
            'wide_return': wide_ret
        })
        
        # Fix logic for correct accumulation
        results[-1]['umaren_return'] = umaren_ret
        results[-1]['wide_return'] = wide_ret

    res_df = pd.DataFrame(results)
    if len(res_df) == 0:
        return
        
    total_bets = res_df['num_pairs'].sum() * 100
    u_ret = res_df['umaren_return'].sum() * 100
    w_ret = res_df['wide_return'].sum() * 100
    
    u_roi = u_ret / total_bets * 100
    w_roi = w_ret / total_bets * 100
    
    marker_u = ' <--!' if u_roi >= 100 else ''
    marker_w = ' <--!' if w_roi >= 100 else ''
    
    print(f"{strategy_name:50s}: Bets={int(total_bets/100):5d}, Umaren ROI={u_roi:5.1f}%{marker_u}, Wide ROI={w_roi:5.1f}%{marker_w}")

print("\n=== 1. Top 3 BOX + Odds Filter (Low odds exclusion) ===")
# Note: we can't filter by pair odds easily because we don't have historical pair odds database.
# We will simulate "Odds Filter" by proxy: "Don't bet if sum of single odds is too low" or similar, 
# but for now let's just use the Top 3 BOX base and assume we buy all.
# Actually user asked for odds filter. Without pair odds, we can try filtering by 
# individual horse odds (e.g. don't buy if favorite odds < 1.5).

# Test: Top 3 BOX, but skip race if Top 1 horse odds < Threshold
for threshold in [1.5, 2.0, 2.5, 3.0, 5.0]:
    def strategy(race_df):
        top1_odds = race_df.iloc[0]['odds']
        if top1_odds < threshold:
            return []
        
        horses = race_df.iloc[:3]['horse_number'].astype(int).tolist()
        if len(horses) < 2: return []
        return [tuple(sorted(p)) for p in combinations(horses, 2)]
        
    run_simulation(f"Top 3 BOX (Skip if Top1 Odds < {threshold})", strategy)


print("\n=== 2. Top 1 Nagashi + EV Filter (Partner EV > X) ===")
# Anchor: Top 1 horse
# Partners: Any horse with EV >= X (limit to top 5 rank to avoid too many)

for min_ev in [0.8, 1.0, 1.2, 1.5]:
    def strategy(race_df):
        anchor = race_df.iloc[0]['horse_number']
        
        # Partners: Rank 2-10, but must have EV >= min_ev
        candidates = race_df.iloc[1:10] # Candidates from rank 2 to 10
        partners = candidates[candidates['ev'] >= min_ev]['horse_number'].astype(int).tolist()
        
        if not partners:
            return []
            
        return [tuple(sorted([int(anchor), p])) for p in partners]

    run_simulation(f"Nagashi Top1 -> Partner EV >= {min_ev}", strategy)

print("\n=== 3. Top 1 Nagashi + High Odds Partners (Ana-nery) ===")
# Anchor: Top 1
# Partners: Rank 2-8, Odds >= 10.0
for min_odds in [10, 15, 20]:
    def strategy(race_df):
        anchor = race_df.iloc[0]['horse_number']
        candidates = race_df.iloc[1:9]
        partners = candidates[candidates['odds'] >= min_odds]['horse_number'].astype(int).tolist()
        if not partners: return []
        return [tuple(sorted([int(anchor), p])) for p in partners]

    run_simulation(f"Nagashi Top1 -> Partner Odds >= {min_odds}", strategy)

print("\nDONE")

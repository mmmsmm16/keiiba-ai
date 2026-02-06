"""
Multi-Bet Type ROI Evaluation for Odds10-Weighted Model
========================================================
Evaluates ROI across all bet types:
- 単勝 (Win)
- 複勝 (Place/Top 3)
- ワイド (Wide)
- 馬連 (Umaren)
- 馬単 (Umatan)
- 三連複 (Sanrenpuku)
- 三連単 (Sanrentan)
"""
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
from scipy.special import softmax
from itertools import combinations, permutations

# Database
engine = create_engine('postgresql://postgres:postgres@db:5432/pckeiba')

# Load payout data
print("Loading payout data...")
payout_query = """
SELECT 
    CONCAT(kaisai_nen, keibajo_code, LPAD(kaisai_kai::text, 2, '0'), 
           LPAD(kaisai_nichime::text, 2, '0'), LPAD(race_bango::text, 2, '0')) as race_id,
    -- Win (単勝)
    haraimodoshi_tansho_1a as win_horse,
    haraimodoshi_tansho_1b as win_payout,
    -- Place (複勝)
    haraimodoshi_fukusho_1a as place_horse_1, haraimodoshi_fukusho_1b as place_payout_1,
    haraimodoshi_fukusho_2a as place_horse_2, haraimodoshi_fukusho_2b as place_payout_2,
    haraimodoshi_fukusho_3a as place_horse_3, haraimodoshi_fukusho_3b as place_payout_3,
    -- Umaren (馬連)
    haraimodoshi_umaren_1a as umaren_pair_1, haraimodoshi_umaren_1b as umaren_payout_1,
    haraimodoshi_umaren_2a as umaren_pair_2, haraimodoshi_umaren_2b as umaren_payout_2,
    haraimodoshi_umaren_3a as umaren_pair_3, haraimodoshi_umaren_3b as umaren_payout_3,
    -- Umatan (馬単)
    haraimodoshi_umatan_1a as umatan_pair_1, haraimodoshi_umatan_1b as umatan_payout_1,
    haraimodoshi_umatan_2a as umatan_pair_2, haraimodoshi_umatan_2b as umatan_payout_2,
    haraimodoshi_umatan_3a as umatan_pair_3, haraimodoshi_umatan_3b as umatan_payout_3,
    -- Wide (ワイド)
    haraimodoshi_wide_1a as wide_pair_1, haraimodoshi_wide_1b as wide_payout_1,
    haraimodoshi_wide_2a as wide_pair_2, haraimodoshi_wide_2b as wide_payout_2,
    haraimodoshi_wide_3a as wide_pair_3, haraimodoshi_wide_3b as wide_payout_3,
    -- Sanrenpuku (三連複)
    haraimodoshi_sanrenpuku_1a as sanrenpuku_trio_1, haraimodoshi_sanrenpuku_1b as sanrenpuku_payout_1,
    haraimodoshi_sanrenpuku_2a as sanrenpuku_trio_2, haraimodoshi_sanrenpuku_2b as sanrenpuku_payout_2,
    haraimodoshi_sanrenpuku_3a as sanrenpuku_trio_3, haraimodoshi_sanrenpuku_3b as sanrenpuku_payout_3,
    -- Sanrentan (三連単)
    haraimodoshi_sanrentan_1a as sanrentan_trio_1, haraimodoshi_sanrentan_1b as sanrentan_payout_1,
    haraimodoshi_sanrentan_2a as sanrentan_trio_2, haraimodoshi_sanrentan_2b as sanrentan_payout_2,
    haraimodoshi_sanrentan_3a as sanrentan_trio_3, haraimodoshi_sanrentan_3b as sanrentan_payout_3
FROM jvd_hr
WHERE kaisai_nen = '2024'
"""
payouts = pd.read_sql(text(payout_query), engine)
print(f"Payout data: {len(payouts)} races")

# --- Parsing Functions ---
def parse_pair(pair_str):
    if pd.isna(pair_str) or len(str(pair_str)) < 4:
        return (0, 0)
    try:
        s = str(pair_str).zfill(4)
        return tuple(sorted([int(s[:2]), int(s[2:4])]))
    except:
        return (0, 0)

def parse_ordered_pair(pair_str):
    if pd.isna(pair_str) or len(str(pair_str)) < 4:
        return (0, 0)
    try:
        s = str(pair_str).zfill(4)
        return (int(s[:2]), int(s[2:4]))
    except:
        return (0, 0)

def parse_trio(trio_str):
    if pd.isna(trio_str) or len(str(trio_str)) < 6:
        return (0, 0, 0)
    try:
        s = str(trio_str).zfill(6)
        return tuple(sorted([int(s[:2]), int(s[2:4]), int(s[4:6])]))
    except:
        return (0, 0, 0)

def parse_ordered_trio(trio_str):
    if pd.isna(trio_str) or len(str(trio_str)) < 6:
        return (0, 0, 0)
    try:
        s = str(trio_str).zfill(6)
        return (int(s[:2]), int(s[2:4]), int(s[4:6]))
    except:
        return (0, 0, 0)

def parse_payout(val):
    try:
        return int(val) / 100  # Convert to odds
    except:
        return 0

# Parse payouts
for i in [1, 2, 3]:
    # Place
    payouts[f'place_h_{i}'] = payouts[f'place_horse_{i}'].apply(lambda x: int(x) if pd.notna(x) else 0)
    payouts[f'place_o_{i}'] = payouts[f'place_payout_{i}'].apply(parse_payout)
    # Umaren
    payouts[f'umaren_p_{i}'] = payouts[f'umaren_pair_{i}'].apply(parse_pair)
    payouts[f'umaren_o_{i}'] = payouts[f'umaren_payout_{i}'].apply(parse_payout)
    # Umatan
    payouts[f'umatan_p_{i}'] = payouts[f'umatan_pair_{i}'].apply(parse_ordered_pair)
    payouts[f'umatan_o_{i}'] = payouts[f'umatan_payout_{i}'].apply(parse_payout)
    # Wide
    payouts[f'wide_p_{i}'] = payouts[f'wide_pair_{i}'].apply(parse_pair)
    payouts[f'wide_o_{i}'] = payouts[f'wide_payout_{i}'].apply(parse_payout)
    # Sanrenpuku
    payouts[f'sanrenpuku_t_{i}'] = payouts[f'sanrenpuku_trio_{i}'].apply(parse_trio)
    payouts[f'sanrenpuku_o_{i}'] = payouts[f'sanrenpuku_payout_{i}'].apply(parse_payout)
    # Sanrentan
    payouts[f'sanrentan_t_{i}'] = payouts[f'sanrentan_trio_{i}'].apply(parse_ordered_trio)
    payouts[f'sanrentan_o_{i}'] = payouts[f'sanrentan_payout_{i}'].apply(parse_payout)

payouts['win_h'] = payouts['win_horse'].apply(lambda x: int(x) if pd.notna(x) else 0)
payouts['win_o'] = payouts['win_payout'].apply(parse_payout)

# --- Load Model & Data ---
print("\nLoading model and data...")
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df_test = df[df['year'] == 2024].copy()
df_test = df_test[(df_test['odds'] > 0) & (df_test['odds'].notna())].reset_index(drop=True)

# Add Undervalued Features (Required for v13 model)
print("Adding undervalued features...")
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

# Load Optimized v13 Model
model_dir = 'models/experiments/exp_lambdarank_v13_optimized'
print(f"Loading model from {model_dir}...")
model = joblib.load(f'{model_dir}/model.pkl')
features = pd.read_csv(f'{model_dir}/features.csv')['0'].tolist()

# Validate features exist
missing_cols = [c for c in features if c not in df_test.columns]
if missing_cols:
    print(f"Warning: Missing columns: {missing_cols}")
    # Fill missing with 0 for safety (though strictly should not happen if logic matches)
    for c in missing_cols:
        df_test[c] = 0

# Predict
scores = model.predict(df_test[features].values)
df_test['pred_score'] = scores
df_test['pred_rank'] = df_test.groupby('race_id')['pred_score'].rank(ascending=False)
df_test = df_test.sort_values(['race_id', 'pred_rank'])

# --- Simulation Functions ---
def simulate_win(df, payouts, top_n=1):
    """Single Win bet on Top N predicted horses"""
    results = []
    for race_id, group in df.groupby('race_id'):
        payout_row = payouts[payouts['race_id'] == race_id]
        if len(payout_row) == 0: continue
        p = payout_row.iloc[0]
        
        top_horses = group.nsmallest(top_n, 'pred_rank')['horse_number'].astype(int).tolist()
        for h in top_horses:
            hit = 1 if h == p['win_h'] else 0
            ret = p['win_o'] if hit else 0
            results.append({'bets': 1, 'return': ret, 'hit': hit})
    
    return pd.DataFrame(results)

def simulate_place(df, payouts, top_n=3):
    """Place bets on Top N predicted horses"""
    results = []
    for race_id, group in df.groupby('race_id'):
        payout_row = payouts[payouts['race_id'] == race_id]
        if len(payout_row) == 0: continue
        p = payout_row.iloc[0]
        
        place_winners = [(p['place_h_1'], p['place_o_1']), 
                         (p['place_h_2'], p['place_o_2']), 
                         (p['place_h_3'], p['place_o_3'])]
        
        top_horses = group.nsmallest(top_n, 'pred_rank')['horse_number'].astype(int).tolist()
        for h in top_horses:
            hit = 0
            ret = 0
            for ph, po in place_winners:
                if h == ph:
                    hit = 1
                    ret = po
                    break
            results.append({'bets': 1, 'return': ret, 'hit': hit})
    
    return pd.DataFrame(results)

def simulate_pair_bet(df, payouts, bet_type='umaren', strategy='box', top_n=3):
    """Pair bets: Umaren, Umatan, Wide"""
    results = []
    for race_id, group in df.groupby('race_id'):
        payout_row = payouts[payouts['race_id'] == race_id]
        if len(payout_row) == 0: continue
        p = payout_row.iloc[0]
        
        top_horses = group.nsmallest(top_n, 'pred_rank')['horse_number'].astype(int).tolist()
        
        if strategy == 'box':
            if bet_type in ['umaren', 'wide']:
                bets = [tuple(sorted(pair)) for pair in combinations(top_horses, 2)]
            else:  # umatan (ordered)
                bets = list(permutations(top_horses, 2))
        elif strategy == 'nagashi':
            anchor = top_horses[0]
            partners = top_horses[1:]
            if bet_type in ['umaren', 'wide']:
                bets = [tuple(sorted([anchor, p])) for p in partners]
            else:
                bets = [(anchor, p) for p in partners]
        
        total_ret = 0
        total_hit = 0
        for bet in bets:
            for i in [1, 2, 3]:
                if bet_type == 'umaren':
                    if bet == p[f'umaren_p_{i}']:
                        total_ret += p[f'umaren_o_{i}']
                        total_hit = 1
                elif bet_type == 'umatan':
                    if bet == p[f'umatan_p_{i}']:
                        total_ret += p[f'umatan_o_{i}']
                        total_hit = 1
                elif bet_type == 'wide':
                    if bet == p[f'wide_p_{i}']:
                        total_ret += p[f'wide_o_{i}']
                        total_hit = 1
        
        results.append({'bets': len(bets), 'return': total_ret, 'hit': total_hit})
    
    return pd.DataFrame(results)

def simulate_trio_bet(df, payouts, bet_type='sanrenpuku', strategy='box', top_n=5):
    """Trio bets: Sanrenpuku, Sanrentan"""
    results = []
    for race_id, group in df.groupby('race_id'):
        payout_row = payouts[payouts['race_id'] == race_id]
        if len(payout_row) == 0: continue
        p = payout_row.iloc[0]
        
        top_horses = group.nsmallest(top_n, 'pred_rank')['horse_number'].astype(int).tolist()
        
        if strategy == 'box':
            if bet_type == 'sanrenpuku':
                bets = [tuple(sorted(trio)) for trio in combinations(top_horses, 3)]
            else:  # sanrentan
                bets = list(permutations(top_horses, 3))
        elif strategy == 'nagashi':  # 1-axis nagashi
            anchor = top_horses[0]
            partners = top_horses[1:]
            if bet_type == 'sanrenpuku':
                bets = [tuple(sorted([anchor] + list(pair))) for pair in combinations(partners, 2)]
            else:
                bets = [(anchor,) + pair for pair in permutations(partners, 2)]
        
        total_ret = 0
        total_hit = 0
        for bet in bets:
            for i in [1, 2, 3]:
                if bet_type == 'sanrenpuku':
                    if bet == p[f'sanrenpuku_t_{i}']:
                        total_ret += p[f'sanrenpuku_o_{i}']
                        total_hit = 1
                elif bet_type == 'sanrentan':
                    if bet == p[f'sanrentan_t_{i}']:
                        total_ret += p[f'sanrentan_o_{i}']
                        total_hit = 1
        
        results.append({'bets': len(bets), 'return': total_ret, 'hit': total_hit})
    
    return pd.DataFrame(results)

def calc_roi(res_df):
    if len(res_df) == 0: return 0, 0, 0
    total_bet = res_df['bets'].sum()
    total_ret = res_df['return'].sum()
    total_hit = res_df['hit'].sum()
    roi = (total_ret / total_bet) * 100 if total_bet > 0 else 0
    hit_rate = (total_hit / len(res_df)) * 100 if len(res_df) > 0 else 0
    return roi, int(total_bet), hit_rate

# --- Run Simulations ---
print("\n" + "="*60)
print("Odds10-Weighted Model - Multi-Bet Type ROI Evaluation")
print("="*60)

print("\n=== 単勝 (Win) ===")
for n in [1, 2, 3]:
    res = simulate_win(df_test, payouts, top_n=n)
    roi, bets, hit = calc_roi(res)
    print(f"Top {n}: Bets={bets}, ROI={roi:.1f}%, Hit={hit:.1f}%")

print("\n=== 複勝 (Place) ===")
for n in [1, 2, 3]:
    res = simulate_place(df_test, payouts, top_n=n)
    roi, bets, hit = calc_roi(res)
    print(f"Top {n}: Bets={bets}, ROI={roi:.1f}%, Hit={hit:.1f}%")

print("\n=== ワイド (Wide) ===")
for strat, n in [('box', 3), ('box', 5), ('nagashi', 5)]:
    res = simulate_pair_bet(df_test, payouts, 'wide', strat, n)
    roi, bets, hit = calc_roi(res)
    print(f"Top {n} {strat.upper()}: Bets={bets}, ROI={roi:.1f}%, Hit={hit:.1f}%")

print("\n=== 馬連 (Umaren) ===")
for strat, n in [('box', 3), ('box', 5), ('nagashi', 5)]:
    res = simulate_pair_bet(df_test, payouts, 'umaren', strat, n)
    roi, bets, hit = calc_roi(res)
    print(f"Top {n} {strat.upper()}: Bets={bets}, ROI={roi:.1f}%, Hit={hit:.1f}%")

print("\n=== 馬単 (Umatan) ===")
for strat, n in [('box', 3), ('nagashi', 5)]:
    res = simulate_pair_bet(df_test, payouts, 'umatan', strat, n)
    roi, bets, hit = calc_roi(res)
    print(f"Top {n} {strat.upper()}: Bets={bets}, ROI={roi:.1f}%, Hit={hit:.1f}%")

print("\n=== 三連複 (Sanrenpuku) ===")
for strat, n in [('box', 5), ('nagashi', 6)]:
    res = simulate_trio_bet(df_test, payouts, 'sanrenpuku', strat, n)
    roi, bets, hit = calc_roi(res)
    print(f"Top {n} {strat.upper()}: Bets={bets}, ROI={roi:.1f}%, Hit={hit:.1f}%")

print("\n=== 三連単 (Sanrentan) ===")
for strat, n in [('box', 3), ('nagashi', 5)]:
    res = simulate_trio_bet(df_test, payouts, 'sanrentan', strat, n)
    roi, bets, hit = calc_roi(res)
    print(f"Top {n} {strat.upper()}: Bets={bets}, ROI={roi:.1f}%, Hit={hit:.1f}%")

print("\nDONE")

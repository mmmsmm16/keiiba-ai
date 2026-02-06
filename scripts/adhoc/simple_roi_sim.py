"""
Simple ROI Simulation - Directly using model predictions and confirmed payouts

No EV filtering, no odds complexity - just pure prediction-based betting
"""
import pandas as pd
import joblib
from sqlalchemy import create_engine

# Load predictions
print("Loading model and data...")
model = joblib.load('models/experiments/optuna_best_full/model.pkl')
df = pd.read_parquet('data/processed/preprocessed_data_v11.parquet')
df['year'] = pd.to_datetime(df['date']).dt.year

# Features
leakage = ['pass_1', 'pass_2', 'pass_3', 'pass_4', 'passing_rank', 'last_3f', 
           'raw_time', 'time_diff', 'margin', 'time', 'popularity', 'odds', 
           'relative_popularity_rank', 'slow_start_recovery', 'track_bias_disadvantage', 
           'outer_frame_disadv', 'wide_run', 'mean_time_diff_5', 'horse_wide_run_rate']
meta = ['race_id', 'horse_number', 'date', 'rank', 'odds_final', 'is_win', 
        'is_top2', 'is_top3', 'year', 'rank_str']
ids = ['horse_id', 'mare_id', 'sire_id', 'jockey_id', 'trainer_id']
feat = [c for c in df.columns if c not in meta + leakage + ids]

# Load payouts
engine = create_engine('postgresql://postgres:postgres@host.docker.internal:5433/pckeiba')
query = """
SELECT 
    kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
    haraimodoshi_tansho_1a as win_horse, haraimodoshi_tansho_1b as win_pay
FROM jvd_hr 
WHERE kaisai_nen IN ('2023', '2024')
"""
df_pay = pd.read_sql(query, engine)
df_pay['race_id'] = df_pay['race_id'].astype(str)
df_pay['win_horse'] = pd.to_numeric(df_pay['win_horse'], errors='coerce').astype('Int64')
df_pay['win_pay'] = pd.to_numeric(df_pay['win_pay'], errors='coerce')

print(f"Payout records: {len(df_pay)}")
print(f"Sample payout: horse={df_pay['win_horse'].iloc[0]}, pay={df_pay['win_pay'].iloc[0]}")

# Build payout dict
payout_dict = {}
for _, row in df_pay.iterrows():
    rid = row['race_id']
    h = row['win_horse']
    p = row['win_pay']
    if pd.notna(h) and pd.notna(p) and h > 0 and p > 0:
        payout_dict[rid] = {int(h): int(p)}

print(f"Payout dict size: {len(payout_dict)}")

results = []

for year in [2023, 2024]:
    print(f"\nProcessing {year}...")
    df_year = df[df['year'] == year].copy()
    
    # Get predictions
    X = df_year[feat].copy()
    for c in X.columns:
        if X[c].dtype.name == 'category' or X[c].dtype == 'object':
            X[c] = X[c].astype('category').cat.codes
        else:
            X[c] = X[c].fillna(-999)
    
    df_year['pred'] = model.predict(X)
    df_year['race_id'] = df_year['race_id'].astype(str)
    
    # Get Top1 per race
    df_year = df_year.sort_values(['race_id', 'pred'], ascending=[True, False])
    df_year['rr'] = df_year.groupby('race_id').cumcount() + 1
    top1 = df_year[df_year['rr'] == 1].copy()
    
    # Simulate betting
    for prob_th in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]:
        bets = top1[top1['pred'] >= prob_th].copy()
        
        # Calculate payouts
        total_bet = 0
        total_payout = 0
        hits = 0
        
        for _, row in bets.iterrows():
            rid = str(row['race_id'])
            hn = int(row['horse_number'])
            
            if rid in payout_dict:
                total_bet += 100
                if hn in payout_dict[rid]:
                    total_payout += payout_dict[rid][hn]
                    hits += 1
        
        if total_bet > 0:
            roi = total_payout / total_bet * 100
            hit_rate = hits / (total_bet / 100) * 100
            results.append({
                'year': year,
                'prob_th': prob_th,
                'races': int(total_bet / 100),
                'hits': hits,
                'hit_rate': hit_rate,
                'roi': roi
            })

# Print results
print("\n" + "=" * 70)
print("Simple ROI Results (Top1 Single Bet, No Filters)")
print("=" * 70)
print(f"{'Year':<6} {'Prob>=':<8} {'Races':<8} {'Hits':<6} {'Hit%':<8} {'ROI%':<8}")
print("-" * 70)

for r in results:
    print(f"{r['year']:<6} {r['prob_th']:<8.2f} {r['races']:<8} {r['hits']:<6} {r['hit_rate']:<7.1f}% {r['roi']:<7.1f}%")

# Best
best = max(results, key=lambda x: x['roi'])
print("\n" + "=" * 70)
print(f"Best: {best['year']} with prob>={best['prob_th']:.2f}")
print(f"  Races: {best['races']}, Hits: {best['hits']}, Hit%: {best['hit_rate']:.1f}%, ROI: {best['roi']:.1f}%")

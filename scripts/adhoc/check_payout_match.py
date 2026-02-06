"""Check payout matching issue"""
import pandas as pd
import joblib
from sqlalchemy import create_engine

# Load predictions
model = joblib.load('models/experiments/optuna_best_full/model.pkl')
df = pd.read_parquet('data/processed/preprocessed_data_v11.parquet')
df['year'] = pd.to_datetime(df['date']).dt.year
df23 = df[df['year'] == 2023].copy()

# Features
leakage = ['pass_1', 'pass_2', 'pass_3', 'pass_4', 'passing_rank', 'last_3f', 
           'raw_time', 'time_diff', 'margin', 'time', 'popularity', 'odds', 
           'relative_popularity_rank', 'slow_start_recovery', 'track_bias_disadvantage', 
           'outer_frame_disadv', 'wide_run', 'mean_time_diff_5', 'horse_wide_run_rate']
meta = ['race_id', 'horse_number', 'date', 'rank', 'odds_final', 'is_win', 
        'is_top2', 'is_top3', 'year', 'rank_str']
ids = ['horse_id', 'mare_id', 'sire_id', 'jockey_id', 'trainer_id']
feat = [c for c in df.columns if c not in meta + leakage + ids]

X = df23[feat].copy()
for c in X.columns:
    if X[c].dtype.name == 'category' or X[c].dtype == 'object':
        X[c] = X[c].astype('category').cat.codes
    else:
        X[c] = X[c].fillna(-999)

df23['pred'] = model.predict(X)
df23 = df23.sort_values(['race_id', 'pred'], ascending=[True, False])
df23['rr'] = df23.groupby('race_id').cumcount() + 1
top1 = df23[df23['rr'] == 1].copy()

# Load payouts
engine = create_engine('postgresql://postgres:postgres@host.docker.internal:5433/pckeiba')
query = """
SELECT 
    kaisai_nen || keibajo_code || kaisai_kai || kaisai_nichime || race_bango as race_id,
    haraimodoshi_tansho_1a as win_horse, haraimodoshi_tansho_1b as win_pay
FROM jvd_hr 
WHERE kaisai_nen = '2023'
"""
df_pay = pd.read_sql(query, engine)
df_pay['race_id'] = df_pay['race_id'].astype(str)

print("=" * 60)
print("Payout Matching Analysis")
print("=" * 60)

# Check race_id format
print(f"\nSample race_ids from predictions: {list(top1['race_id'].astype(str).head(3))}")
print(f"Sample race_ids from payouts: {list(df_pay['race_id'].head(3))}")

# Check match rate
top1['race_id'] = top1['race_id'].astype(str)
matched = top1.merge(df_pay, on='race_id', how='left')
print(f"\nTop1 count: {len(top1)}")
print(f"Payout records: {len(df_pay)}")
print(f"Matched: {matched['win_horse'].notna().sum()}")

# Check if horse_number matches
matched['win_horse'] = pd.to_numeric(matched['win_horse'], errors='coerce')
matched['horse_number'] = matched['horse_number'].astype(int)
matched['is_actual_winner'] = matched['rank'] == 1
matched['payout_shows_winner'] = matched['horse_number'] == matched['win_horse']

print(f"\nActual winners from rank==1: {matched['is_actual_winner'].sum()}")
print(f"Matches with win_horse: {matched['payout_shows_winner'].sum()}")

# Debug sample
print("\n Sample of Top1 predictions vs payouts:")
sample = matched[['race_id', 'horse_number', 'rank', 'win_horse', 'win_pay', 'is_actual_winner', 'payout_shows_winner']].head(10)
print(sample.to_string())

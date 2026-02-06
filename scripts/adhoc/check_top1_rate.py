"""Quick diagnostic: Check Top1 prediction win rate"""
import pandas as pd
import joblib

# Load model and data
model = joblib.load('models/experiments/optuna_best_full/model.pkl')
df = pd.read_parquet('data/processed/preprocessed_data_v11.parquet')
df['year'] = pd.to_datetime(df['date']).dt.year
df23 = df[df['year'] == 2023].copy()

# Leakage cols
leakage = ['pass_1', 'pass_2', 'pass_3', 'pass_4', 'passing_rank', 'last_3f', 
           'raw_time', 'time_diff', 'margin', 'time', 'popularity', 'odds', 
           'relative_popularity_rank', 'slow_start_recovery', 'track_bias_disadvantage', 
           'outer_frame_disadv', 'wide_run', 'mean_time_diff_5', 'horse_wide_run_rate']
meta = ['race_id', 'horse_number', 'date', 'rank', 'odds_final', 'is_win', 
        'is_top2', 'is_top3', 'year', 'rank_str']
ids = ['horse_id', 'mare_id', 'sire_id', 'jockey_id', 'trainer_id']
feat = [c for c in df.columns if c not in meta + leakage + ids]

# Prepare features
X = df23[feat].copy()
for c in X.columns:
    if X[c].dtype.name == 'category' or X[c].dtype == 'object':
        X[c] = X[c].astype('category').cat.codes
    else:
        X[c] = X[c].fillna(-999)

# Predict
df23['pred'] = model.predict(X)

# Get Top1 per race
df23 = df23.sort_values(['race_id', 'pred'], ascending=[True, False])
df23['rr'] = df23.groupby('race_id').cumcount() + 1
t1 = df23[df23['rr'] == 1]

print("=" * 60)
print("Top1 Prediction Win Rate Analysis")
print("=" * 60)

n_races = len(t1)
n_wins = (t1['rank'] == 1).sum()
win_rate = (t1['rank'] == 1).mean() * 100

print(f"Total races (2023): {n_races}")
print(f"Top1 predictions that won: {n_wins}")
print(f"Top1 win rate: {win_rate:.1f}%")

# Compare with random baseline
avg_field_size = df23.groupby('race_id').size().mean()
random_win_rate = 100 / avg_field_size
print(f"\nBaseline (random) win rate: {random_win_rate:.1f}%")
print(f"Improvement over random: {win_rate / random_win_rate:.2f}x")

# Top-N analysis
for n in [1, 2, 3, 5]:
    topn = df23[df23['rr'] <= n]
    topn_wins = (topn['rank'] == 1).sum()
    topn_rate = topn_wins / n_races * 100
    print(f"\nTop{n} contains winner: {topn_rate:.1f}%")

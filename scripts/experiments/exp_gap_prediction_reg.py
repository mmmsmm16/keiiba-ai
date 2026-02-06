
"""
Regularized Gap Prediction Model Experiment
===========================================
Concept:
Predict the "Gap" between Popularity Rank and Actual Rank, but with REGULARIZATION.

Changes from v1:
1. Label Clipping:
   - Target = max(0, min(Gap, 5))
   - Prevents model from obsessing over "Rank 18 -> Rank 1" (Gap 17) impossible shots.
   - Focuses on "Rank 6 -> Rank 1" (Gap 5) realistic holes.

2. Log Weighting:
   - Weight = log1p(Odds)
   - Reduces the impact of extreme odds (e.g. 100x vs 10x is not 10x more important, but log scale).

Goal:
Find undervalued horses with a better balance of Hit Rate and ROI.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
from sklearn.metrics import ndcg_score

# Load data
print("Loading data...")
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df = df[(df['odds'] > 0) & (df['odds'].notna())].reset_index(drop=True)

# --- Feature Engineering ---
print("Injecting features...")
df['odds_rank'] = df.groupby('race_id')['odds'].rank(ascending=True)

if 'relative_horse_elo_z' in df.columns:
    df['elo_rank'] = df.groupby('race_id')['relative_horse_elo_z'].rank(ascending=False)
    df['odds_rank_vs_elo'] = df['odds_rank'] - df['elo_rank']
else:
    df['odds_rank_vs_elo'] = 0

df['is_high_odds'] = (df['odds'] >= 10).astype(int)
df['is_mid_odds'] = ((df['odds'] >= 5) & (df['odds'] < 10)).astype(int)

# Use numeric features except yoso and leaks
# CONFIRMED LEAKS:
#   - yoso_rank_diff: Contains actual finishing position (実着順)
#   - late_charge, corner_position_change, makuri_positions: Current race corner data
exclude = ['rank', 'date', 'race_id', 'horse_id', 'jockey_id', 'trainer_id', 'target', 'year', 'odds', 
           'yoso_juni', 'yoso_p_std', 'yoso_p_mean', 
           'time', 'pop_rank', 'gap', 'target_label', 'vote_count',
           'corner_position_change', 'makuri_positions', 'yoso_time_diff',
           'yoso_rank_diff',  # CRITICAL: Contains actual rank!
           'late_charge',     # Current race 3-4 corner
           ]
all_features = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

print(f"Features: {len(all_features)}")

# --- Target Generation (Improved: Top 5 Only) ---
df['pop_rank'] = df.groupby('race_id')['odds'].rank(ascending=True)
df['gap'] = df['pop_rank'] - df['rank']

# NEW: Only reward horses that:
# 1. Finished in Top 5 (rank <= 5) - these are the ones that matter for betting
# 2. Beat their popularity (gap > 0)
# This ignores "18番人気 → 10着" (Gap=+8 but useless) 
# and rewards "8番人気 → 3着" (Gap=+5, valuable)
CLIP_THRESHOLD = 3
TOP_N_CUTOFF = 5

def compute_target(row):
    if row['rank'] <= TOP_N_CUTOFF and row['gap'] > 0:
        return int(min(row['gap'], CLIP_THRESHOLD))
    return 0

df['target_label'] = df.apply(compute_target, axis=1)
print(f"Target distribution: {df['target_label'].value_counts().sort_index().to_dict()}")


# --- Split ---
df_train = df[df['year'] <= 2022].copy()
df_valid = df[df['year'] == 2023].copy()
df_test = df[df['year'] == 2024].copy()

# Sort for LambdaRank
df_train = df_train.sort_values('race_id')
df_valid = df_valid.sort_values('race_id')
df_test = df_test.sort_values('race_id')

train_groups = df_train.groupby('race_id').size().values
valid_groups = df_valid.groupby('race_id').size().values

X_train = df_train[all_features].values
X_valid = df_valid[all_features].values
X_test = df_test[all_features].values

y_train = df_train['target_label'].values
y_valid = df_valid['target_label'].values

# Weights: Uniform (to avoid extreme longshot bias)
# Previous: log1p(odds) - too aggressive on longshots
# Now: Uniform - all gap levels treated equally
w_train = np.ones(len(y_train))  # Uniform weights

train_data = lgb.Dataset(X_train, label=y_train, group=train_groups, weight=w_train, feature_name=all_features)
valid_data = lgb.Dataset(X_valid, label=y_valid, group=valid_groups, reference=train_data, feature_name=all_features)

# --- Training ---
print("\nTraining Regularized Gap Prediction Model (Leak-Free)...")
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 3, 5],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.02, 
    'n_estimators': 1000,
    'seed': 42,
    'verbosity': 1,  # Verbose output
    'label_gain': [i for i in range(CLIP_THRESHOLD + 1)] # 0,1,2,3
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=params['n_estimators'],
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)]  # Log every 50 rounds
)

# --- Evaluation ---
print("\n=== Evaluation (Regularized Gap) ===")
scores = model.predict(X_test)
df_test['pred_score'] = scores
df_test['pred_rank'] = df_test.groupby('race_id')['pred_score'].rank(ascending=False)

# 1. Top 1 Stats
top1 = df_test[df_test['pred_rank'] == 1]
wins = top1[top1['rank'] == 1]
places = top1[top1['rank'] <= 3]

print("\nTop 1 Predicted Stats:")
print(top1[['odds', 'rank', 'gap', 'pop_rank']].describe())

roi_win = wins['odds'].sum() / len(top1) * 100
hit_rate_win = len(wins) / len(top1) * 100
hit_rate_place = len(places) / len(top1) * 100

print(f"Top 1 Win ROI: {roi_win:.2f}% (Hit Rate: {hit_rate_win:.1f}%)")
print(f"Top 1 Place Hit Rate: {hit_rate_place:.1f}%")

# 2. Hole Finding (Pop > 5)
holes = top1[top1['pop_rank'] > 5]
print(f"\nTop 1 Predictions that were unpopular (Pop > 5): {len(holes)} bets ({len(holes)/len(top1)*100:.1f}%)")
if len(holes) > 0:
    hole_wins = holes[holes['rank'] == 1]
    hole_roi = hole_wins['odds'].sum() / len(holes) * 100
    print(f"Hole Win ROI: {hole_roi:.2f}% (Hit Rate: {len(hole_wins)/len(holes)*100:.1f}%)")

# Save
OUTPUT_DIR = 'models/experiments/exp_gap_prediction_reg'
os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump(model, f'{OUTPUT_DIR}/model.pkl')
pd.DataFrame({'0': all_features}).to_csv(f'{OUTPUT_DIR}/features.csv', index=False)
print(f"Model saved to {OUTPUT_DIR}")

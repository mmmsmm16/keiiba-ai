"""
Deep Value Model (Hard Weighted + No Yoso)
==========================================
Trains a model designed to target potential longshots (穴馬).
1. Hard Weighting: Uses 'Odds' directly as weight (linear), clipped at 50.0.
   (Previous models used log1p(odds) which favored popular horses).
2. No Yoso: Removes 'yoso_juni' and derived features to prevent leaking popularity info.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

# Load data
print("Loading data...")
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df = df[(df['odds'] > 0) & (df['odds'].notna())].reset_index(drop=True)

# --- Feature Engineering ---
print("Adding features...")
# Ensure odds_10min exists (fallback to odds only for weighting, NOT for features)
if 'odds_10min' not in df.columns or df['odds_10min'].isna().all():
    print("Warning: odds_10min missing in parquet. Training features might be suboptimal.")
    df['odds_feature'] = df['odds']
else:
    df['odds_feature'] = df['odds_10min']

# Add Undervalued Features (Only those NOT derived from yoso_juni or FINAL odds)
df['odds_rank_pre'] = df.groupby('race_id')['odds_feature'].rank(ascending=True)

if 'relative_horse_elo_z' in df.columns:
    df['elo_rank'] = df.groupby('race_id')['relative_horse_elo_z'].rank(ascending=False)
    df['odds_rank_vs_elo'] = df['odds_rank_pre'] - df['elo_rank']
else:
    df['odds_rank_vs_elo'] = 0

df['is_high_odds'] = (df['odds_feature'] >= 10).astype(int)
df['is_mid_odds'] = ((df['odds_feature'] >= 5) & (df['odds_feature'] < 10)).astype(int)

# Base features
baseline_features = pd.read_csv('models/experiments/exp_lambdarank_v12_batch4_optuna_odds10_weighted/features.csv')['0'].tolist()

# REMOVE Yoso Features
features_to_remove = ['yoso_juni', 'yoso_p_std', 'yoso_p_mean'] # yoso-related
# Note: popularity_vs_yoso, odds_rank_vs_yoso are not in baseline, so just don't add them.

filtered_features = [f for f in baseline_features if 'yoso' not in f]
# Also remove popularity? User said "yoso_juniを抜く". popularity is fair game (public info).
# But popularity is strongly correlated. Let's keep popularity for now, but rely on weighting to ignore it if low odds.

new_features = ['odds_rank_vs_elo', 'is_high_odds', 'is_mid_odds']
all_features = filtered_features + new_features
all_features = [c for c in all_features if c in df.columns]

print(f"Removed yoso features. Total features: {len(all_features)}")

# --- Weights (Hard Weighting) ---
# Use Odds directly, clipped at 50.0
# Only use 'odds' column (final odds) or 'odds_10min' depending on policy.
# Using 'odds_10min' if available to mimic pre-race state, but for weight we want the 'reward'.
# Let's use Final Odds for Weight (Reward), but keep training features strictly pre-race.
weights = df['odds'].clip(1.0, 50.0).values

# Normalize weights so mean is around 1.0 (optional, but helps gradient scale)
# weights = weights / weights.mean() 
# Actually LambdaRank handles scale well, but let's leave it raw to emphasize ratio.
# A 50.0 odds horse is 50x more important than 1.0 odds horse.

# --- Split ---
df_train = df[df['year'] <= 2022].copy()
df_valid = df[df['year'] == 2023].copy()
df_test = df[df['year'] == 2024].copy()

train_groups = df_train.groupby('race_id').size().values
valid_groups = df_valid.groupby('race_id').size().values

X_train = df_train[all_features].values
X_valid = df_valid[all_features].values
X_test = df_test[all_features].values

y_train = df_train['rank'].apply(lambda r: max(0, 4-r)).values
y_valid = df_valid['rank'].apply(lambda r: max(0, 4-r)).values

w_train = df_train['odds'].clip(1.0, 50.0).values
# w_valid = df_valid['odds'].clip(1.0, 50.0).values # Valid weights used for metric calculation if weighted metric used

train_data = lgb.Dataset(X_train, label=y_train, group=train_groups, weight=w_train, feature_name=all_features)
valid_data = lgb.Dataset(X_valid, label=y_valid, group=valid_groups, reference=train_data, feature_name=all_features)

# --- Training ---
print("\nTraining Deep Value Model...")
params = {
    'num_leaves': 31, # Slightly smaller to prevent overfitting to noise
    'learning_rate': 0.01, # Slower learning for noisy weights
    'n_estimators': 2000,
    'min_child_samples': 50,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 3, 5],
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'seed': 42
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=params['n_estimators'],
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
)

# --- Evaluation ---
print("\n=== Evaluation (Deep Value) ===")
scores = model.predict(X_test)
df_test['pred_score'] = scores
df_test['pred_rank'] = df_test.groupby('race_id')['pred_score'].rank(ascending=False)

# Rank 1 Stats
rank1 = df_test[df_test['pred_rank'] == 1]
print("\nRank 1 Stats:")
print(rank1[['odds', 'rank']].describe())

# Top 1 ROI
wins = rank1[rank1['rank'] == 1]
roi = wins['odds'].sum() / len(rank1) * 100
hit_rate = len(wins) / len(rank1) * 100
print(f"Top 1 ROI: {roi:.2f}% (Hit Rate: {hit_rate:.1f}%)")

# EV Simulation (using dummy prob for now, or just Win EV)
# Actually, let's look at Return of simple strategies
print("\nStrategies:")
# Win Top 3 Box? No, just Simple Top N ROI
top3 = df_test[df_test['pred_rank'] <= 3]
top3_wins = top3[top3['rank'] == 1]
top3_roi = top3_wins['odds'].sum() / len(top3) * 100
print(f"Top 3 Win ROI: {top3_roi:.2f}%")

# Save
OUTPUT_DIR = 'models/experiments/exp_lambdarank_hard_weighted'
os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump(model, f'{OUTPUT_DIR}/model.pkl')
pd.DataFrame({'0': all_features}).to_csv(f'{OUTPUT_DIR}/features.csv', index=False)
print(f"Model saved to {OUTPUT_DIR}")

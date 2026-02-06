"""
Evaluate Optimized v13 Model
============================
Retrains v13 model with optimized parameters and evaluates ROI.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import json
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

# Load best params
with open('models/experiments/exp_lambdarank_v13_optuna/best_params.json', 'r') as f:
    best_params = json.load(f)

print("Best Params:", best_params)

# Load data
print("Loading data...")
df = pd.read_parquet('data/processed/preprocessed_data_v12.parquet')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df = df[(df['odds'] > 0) & (df['odds'].notna())].reset_index(drop=True)

# --- Add Undervalued Features ---
print("Adding undervalued features...")
df['yoso_juni_num'] = pd.to_numeric(df['yoso_juni'], errors='coerce').fillna(8)
df['popularity_num'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(8)
df['popularity_vs_yoso'] = df['popularity_num'] - df['yoso_juni_num']
df['odds_rank'] = df.groupby('race_id')['odds'].rank(ascending=True)
df['odds_rank_vs_yoso'] = df['odds_rank'] - df['yoso_juni_num']

if 'relative_horse_elo_z' in df.columns:
    df['elo_rank'] = df.groupby('race_id')['relative_horse_elo_z'].rank(ascending=False)
    df['odds_rank_vs_elo'] = df['odds_rank'] - df['elo_rank']
else:
    df['odds_rank_vs_elo'] = 0

df['is_high_odds'] = (df['odds'] >= 10).astype(int)
df['is_mid_odds'] = ((df['odds'] >= 5) & (df['odds'] < 10)).astype(int)

# --- Features ---
baseline_features = pd.read_csv('models/experiments/exp_lambdarank_v12_batch4_optuna_odds10_weighted/features.csv')['0'].tolist()
new_features = ['popularity_vs_yoso', 'odds_rank_vs_yoso', 'odds_rank_vs_elo', 'is_high_odds', 'is_mid_odds']
all_features = baseline_features + new_features
all_features = [c for c in all_features if c in df.columns]

# --- Split ---
df_train = df[df['year'] <= 2022].copy()
df_valid = df[df['year'] == 2023].copy()
df_test = df[df['year'] == 2024].copy()

# Weights
train_weights = np.log1p(df_train['odds_10min'].fillna(df_train['odds']).clip(lower=1.01)).values * 0.786
train_weights = np.clip(train_weights, 0, 51.0)

# Datasets
train_groups = df_train.groupby('race_id').size().values
valid_groups = df_valid.groupby('race_id').size().values
X_train = df_train[all_features].values
X_valid = df_valid[all_features].values
X_test = df_test[all_features].values
y_train = df_train['rank'].apply(lambda r: max(0, 4-r)).values
y_valid = df_valid['rank'].apply(lambda r: max(0, 4-r)).values

train_data = lgb.Dataset(X_train, label=y_train, group=train_groups, weight=train_weights, feature_name=all_features)
valid_data = lgb.Dataset(X_valid, label=y_valid, group=valid_groups, reference=train_data, feature_name=all_features)

# --- Train ---
print("\nTraining optimized model...")
params = best_params.copy()
params.update({
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 3, 5],
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'seed': 42
})

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
)

# --- Evaluate ---
print("\n=== Evaluation (Optimized) ===")
scores = model.predict(X_test)
df_test['pred_score'] = scores
df_test['target'] = (df_test['rank'] == 1).astype(int)
df_test['pred_prob'] = df_test.groupby('race_id')['pred_score'].transform(lambda x: softmax(x.values))

auc = roc_auc_score(df_test['target'], df_test['pred_prob'])
print(f"Test AUC: {auc:.4f}")

# Top 1 ROI
df_test['pred_rank'] = df_test.groupby('race_id')['pred_score'].rank(ascending=False)
top1 = df_test[df_test['pred_rank'] == 1]
wins = top1[top1['target'] == 1]
roi = wins['odds'].sum() / len(top1) * 100
print(f"Top 1 ROI: {roi:.2f}% (Hit Rate: {len(wins)/len(top1)*100:.1f}%)")

# EV Filter
print("\nEV Filter Check:")
df_test['ev'] = df_test['pred_prob'] * df_test['odds']
for ev_thr in [2.0, 2.3, 2.5]:
    for max_odds in [15, 20]:
        bets = df_test[(df_test['ev'] >= ev_thr) & (df_test['odds'] <= max_odds)]
        if len(bets) < 10: continue
        w = bets[bets['target'] == 1]
        roi = w['odds'].sum() / len(bets) * 100
        print(f"EV >= {ev_thr}, Odds <= {max_odds}: Bets={len(bets)}, ROI={roi:.1f}%")

# Save
OUTPUT_DIR = 'models/experiments/exp_lambdarank_v13_optimized'
os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump(model, f'{OUTPUT_DIR}/model.pkl')
pd.DataFrame({'0': all_features}).to_csv(f'{OUTPUT_DIR}/features.csv', index=False)
print(f"Model saved to {OUTPUT_DIR}")

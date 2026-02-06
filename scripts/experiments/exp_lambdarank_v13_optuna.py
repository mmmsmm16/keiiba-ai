"""
Optuna Optimization for v13 Undervalued Model
=============================================
Optimizes hyperparameters for the LambdaRank model with updated features.
Focus on learning rate, num_leaves, and regularization to address early stopping.
"""
import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib
import json

# Set random seed
np.random.seed(42)

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

# --- Prepare Features ---
baseline_features = pd.read_csv('models/experiments/exp_lambdarank_v12_batch4_optuna_odds10_weighted/features.csv')['0'].tolist()
new_features = ['popularity_vs_yoso', 'odds_rank_vs_yoso', 'odds_rank_vs_elo', 'is_high_odds', 'is_mid_odds']
all_features = baseline_features + new_features
all_features = [c for c in all_features if c in df.columns]
print(f"Total features: {len(all_features)}")

# --- Split Data ---
df_train = df[df['year'] <= 2022].copy()
df_valid = df[df['year'] == 2023].copy()
# df_test = df[df['year'] == 2024].copy() # Not used for optuna

# --- Weights (Odds10-Weighted) ---
train_weights = np.log1p(df_train['odds_10min'].fillna(df_train['odds']).clip(lower=1.01)).values * 0.786
train_weights = np.clip(train_weights, 0, 51.0)

# --- Datasets ---
train_groups = df_train.groupby('race_id').size().values
valid_groups = df_valid.groupby('race_id').size().values

X_train = df_train[all_features].values
X_valid = df_valid[all_features].values
y_train = df_train['rank'].apply(lambda r: max(0, 4-r)).values
y_valid = df_valid['rank'].apply(lambda r: max(0, 4-r)).values

train_data = lgb.Dataset(X_train, label=y_train, group=train_groups, weight=train_weights, feature_name=all_features)
valid_data = lgb.Dataset(X_valid, label=y_valid, group=valid_groups, reference=train_data, feature_name=all_features)

def objective(trial):
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'n_estimators': 1000,
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    # Pruning callback
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "ndcg@1", valid_name="valid_0")
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=params['n_estimators'],
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0), pruning_callback]
    )
    
    # Return best ndcg@1
    # best_score is in model.best_score or can be retrieved
    # LightGBM records dict: {'valid_0': {'ndcg@1': ...}}
    # But optuna works better if we just return the final best metric
    return list(model.best_score['valid_0'].values())[0]

# --- Run Optimization ---
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, timeout=3600)

print("\nBest trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Save best params
OUTPUT_DIR = 'models/experiments/exp_lambdarank_v13_optuna'
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(f'{OUTPUT_DIR}/best_params.json', 'w') as f:
    json.dump(trial.params, f, indent=4)

print(f"\nBest params saved to {OUTPUT_DIR}")

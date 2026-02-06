import joblib
import pandas as pd
import lightgbm as lgb
import os
import numpy as np

# Configuration
CACHE_PATH = "data/cache/jra_base/advanced.parquet"
BINARY_MODEL_PATH = "models/binary_no_odds.pkl"
RANKER_MODEL_PATH = "models/ranker_no_odds.pkl"

# LambdaRank Params (Optimized)
LGB_PARAMS_RANK = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 3, 5],
    'boosting_type': 'gbdt',
    'learning_rate': 0.044449,
    'num_leaves': 65,
    'min_data_in_leaf': 83,
    'feature_fraction': 0.860430,
    'bagging_fraction': 0.787839,
    'bagging_freq': 5,
    'lambda_l1': 0.010220,
    'lambda_l2': 0.000172,
    'random_state': 42,
    'verbose': -1
}

def train():
    print("Loading data...")
    df = pd.read_parquet(CACHE_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # Train on 2020-01-01 to 2025-12-14
    train_df = df[(df['date'] >= '2020-01-01') & (df['date'] <= '2025-12-14')].copy()
    
    # Label for LambdaRank
    train_df['relevance'] = train_df['rank'].apply(lambda r: 3 if r == 1 else (2 if r == 2 else (1 if r == 3 else 0)))
    
    # Get feature list from binary model
    print(f"Loading Binary model features...")
    binary_data = joblib.load(BINARY_MODEL_PATH)
    feature_cols = binary_data['feature_cols']
    
    print("Preparing features...")
    X_train = train_df[feature_cols].copy()
    for col in X_train.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_train = X_train.fillna(0).astype('float32')
    
    q_train = train_df.groupby('race_id')['horse_id'].count().values
    
    print("Training LambdaRank model...")
    rank_model = lgb.LGBMRanker(**LGB_PARAMS_RANK)
    rank_model.fit(X_train, train_df['relevance'], group=q_train)
    
    print(f"Saving model to {RANKER_MODEL_PATH}...")
    os.makedirs("models", exist_ok=True)
    joblib.dump({'model': rank_model, 'feature_cols': feature_cols}, RANKER_MODEL_PATH)
    print("Success.")

if __name__ == "__main__":
    train()

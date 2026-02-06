"""
Ensemble Backtest: Binary Classification (Top3) + LambdaRank
Objective: Evaluate if combining these two models improves performance (ROI, Win Rate).
"""
import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CACHE_PATH = "data/cache/jra_base/advanced.parquet"
BINARY_MODEL_PATH = "models/binary_no_odds.pkl"

# LambdaRank Params (from run_jra_pipeline_backtest.py)
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

def get_features(df, feature_cols):
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0).astype('float32')
    return X

def normalize_series(s):
    if s.max() == s.min():
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def run_ensemble_backtest():
    # 1. Load Data
    logger.info("Loading data...")
    df = pd.read_parquet(CACHE_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # Split
    train_df = df[(df['date'] >= '2020-01-01') & (df['date'] < '2024-01-01')].copy()
    test_df = df[(df['date'] >= '2024-01-01') & (df['date'] < '2025-01-01')].copy()
    
    # 2. Load Binary Model
    logger.info(f"Loading Binary model from {BINARY_MODEL_PATH}")
    binary_data = joblib.load(BINARY_MODEL_PATH)
    binary_model = binary_data['model']
    feature_cols = binary_data['feature_cols']
    
    # 3. Train LambdaRank Model
    logger.info("Training LambdaRank model...")
    # Target for LambdaRank: 1.0 / rank (or just -rank, but 1/rank is common)
    # Actually, standard is relevance score (3: 1st, 2: 2nd, 1: 3rd, 0: others)
    train_df['relevance'] = train_df['rank'].apply(lambda r: 3 if r == 1 else (2 if r == 2 else (1 if r == 3 else 0)))
    
    X_train = get_features(train_df, feature_cols)
    q_train = train_df.groupby('race_id')['horse_id'].count().values
    
    rank_model = lgb.LGBMRanker(**LGB_PARAMS_RANK)
    rank_model.fit(X_train, train_df['relevance'], group=q_train)
    
    # 4. Predict on Test Set
    logger.info("Predicting...")
    X_test = get_features(test_df, feature_cols)
    
    test_df['prob_binary'] = binary_model.predict_proba(X_test)[:, 1]
    test_df['score_rank'] = rank_model.predict(X_test)
    
    # 5. Normalize within Race
    logger.info("Normalizing scores within races...")
    test_df['norm_binary'] = test_df.groupby('race_id')['prob_binary'].transform(normalize_series)
    test_df['norm_rank'] = test_df.groupby('race_id')['score_rank'].transform(normalize_series)
    
    # 6. Ensemble & Evaluation
    weights = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0] # 1.0 means Binary only, 0.0 means Rank only
    
    results = []
    for w in weights:
        test_df['ensemble_score'] = w * test_df['norm_binary'] + (1 - w) * test_df['norm_rank']
        test_df['pred_rank'] = test_df.groupby('race_id')['ensemble_score'].rank(ascending=False, method='first')
        
        # Calculate Metrics
        top1 = test_df[test_df['pred_rank'] == 1].copy()
        top1_valid = top1[top1['odds'] > 0]
        
        n_races = len(top1_valid)
        if n_races == 0: continue
        
        n_wins = (top1_valid['rank'] == 1).sum()
        win_rate = n_wins / n_races * 100
        
        invest = n_races * 100
        payout = (top1_valid[top1_valid['rank'] == 1]['odds'] * 100).sum()
        roi = payout / invest * 100
        
        results.append({
            'Weight_Binary': w,
            'Weight_Rank': 1-w,
            'Win_Rate': win_rate,
            'ROI': roi,
            'N': n_races
        })
        logger.info(f"Weight Binary={w:.1f}: Win Rate={win_rate:.1f}%, ROI={roi:.1f}%")

    # Output table
    res_df = pd.DataFrame(results)
    print("\n=== Ensemble Backtest Summary (2024 JRA) ===")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    run_ensemble_backtest()

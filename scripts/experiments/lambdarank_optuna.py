"""
LambdaRank Hyperparameter Optimization with Optuna
==================================================
Optimizes LightGBM LambdaRank hyperparameters to maximize NDCG.
Now includes checks for low-importance features pruning.

Usage:
  python scripts/experiments/lambdarank_optuna.py
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import optuna
from sklearn.metrics import ndcg_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
TARGET_PATH = "data/temp_t2/T2_targets.parquet"
BASE_MODEL_PATH = "models/experiments/exp_t2_refined_v3/model.pkl"
OUTPUT_PARAMS_PATH = "models/experiments/exp_lambdarank/best_params.json"

def load_data():
    logger.info("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    targets = pd.read_parquet(TARGET_PATH)
    
    df['race_id'] = df['race_id'].astype(str)
    targets['race_id'] = targets['race_id'].astype(str)
    
    df = df.merge(targets[['race_id', 'horse_number', 'rank']], 
                  on=['race_id', 'horse_number'], how='left')
    df['date'] = pd.to_datetime(df['date'])
    
    # Relevance for LambdaRank
    df['relevance'] = 0
    df.loc[df['rank'] == 1, 'relevance'] = 3
    df.loc[df['rank'] == 2, 'relevance'] = 2
    df.loc[df['rank'] == 3, 'relevance'] = 1
    
    return df

def prepare_lgb_dataset(df, feature_cols):
    df = df.sort_values('race_id')
    X = df[feature_cols].copy()
    y = df['relevance'].values
    group = df.groupby('race_id').size().values
    
    for c in X.columns:
        if X[c].dtype == 'object' or X[c].dtype.name == 'category':
            X[c] = X[c].astype('category').cat.codes
        X[c] = X[c].fillna(-999)
        
    ds = lgb.Dataset(X, label=y, group=group)
    return ds

def objective(trial):
    # Hyperparameters
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        'num_threads': 4, # Parallelism
        'feature_pre_filter': False,
        
        # Tuning range
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 300),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'max_depth': trial.suggest_int('max_depth', -1, 15),
        
        # LambdaRank specific
        # label_gain: Gain for relevance 0, 1, 2, 3
        # Standard: 0, 1, 3, 7. Let's tune higher gains for rank 1
        # 'label_gain': [0, 1, 3, 7] # Fixed for now or create custom logic
    }
    
    # Custom label gain tuning (categorical choice)
    gain_type = trial.suggest_categorical('gain_type', ['standard', 'heavy_win', 'balanced'])
    if gain_type == 'standard':
        params['label_gain'] = [0, 1, 3, 7]
    elif gain_type == 'heavy_win':
        params['label_gain'] = [0, 1, 3, 15] # Emphasize 1st place
    elif gain_type == 'balanced':
        params['label_gain'] = [0, 1, 2, 4] # Less gap
        
    # Pruning callback
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'ndcg@3')
    
    # Train
    model = lgb.train(
        params,
        train_ds,
        num_boost_round=1000,
        valid_sets=[valid_ds],
        callbacks=[lgb.early_stopping(50), pruning_callback]
    )
    
    # Return best score (NDCG@3)
    # LightGBM records metrics in model.best_score['valid_0']['ndcg@3']
    return model.best_score['valid_0']['ndcg@3']

def run_optimization():
    global train_ds, valid_ds # For objective
    
    logger.info("=" * 60)
    logger.info("LambdaRank Optuna Optimization")
    logger.info("=" * 60)
    
    df = load_data()
    
    # Features
    base_model = joblib.load(BASE_MODEL_PATH)
    feature_names = base_model.feature_name()
    exclude_cols = ['race_id', 'horse_number', 'date', 'rank', 'relevance', 'rank_str']
    feature_cols = [c for c in df.columns if c in feature_names and c not in exclude_cols]
    
    # Split
    df_train = df[df['date'].dt.year <= 2022].copy()
    df_valid = df[df['date'].dt.year == 2023].copy()
    # Test not used for optimization
    
    logger.info(f"Train: {len(df_train)}")
    logger.info(f"Valid: {len(df_valid)}")
    
    train_ds = prepare_lgb_dataset(df_train, feature_cols)
    valid_ds = prepare_lgb_dataset(df_valid, feature_cols)
    
    # Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50) # 50 trials
    
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  NDCG@3: {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
        
    # Save best params
    import json
    os.makedirs(os.path.dirname(OUTPUT_PARAMS_PATH), exist_ok=True)
    with open(OUTPUT_PARAMS_PATH, 'w') as f:
        json.dump(trial.params, f, indent=4)
    logger.info(f"Saved best params to {OUTPUT_PARAMS_PATH}")

if __name__ == "__main__":
    run_optimization()

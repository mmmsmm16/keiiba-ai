"""
LightGBM Hyperparameter Optimization with Optuna
Uses cross-validation on historical data to find optimal parameters.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import GroupKFold
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load cached data
CACHE_PATH = "data/cache/jra_base/advanced.parquet"

def load_data():
    """Load preprocessed data"""
    logger.info("Loading data...")
    df = pd.read_parquet(CACHE_PATH)
    
    # Filter to 2020-2024 for optimization (use 2025 for final validation)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= '2020-01-01') & (df['date'] < '2025-01-01')]
    
    logger.info(f"Data loaded: {len(df)} records, {df['date'].min()} to {df['date'].max()}")
    return df

def get_features(df):
    """Get feature matrix"""
    # Drop non-feature columns
    drop_cols = [
        'race_id', 'horse_id', 'horse_number', 'date', 'rank', 'time', 'target',
        'mare_id', 'sire_id', 'trainer_id', 'jockey_id', 'bms_id',
        'odds', 'popularity', 'owner_id', 'breeder_id',
        'lag1_odds', 'lag1_popularity',  # No odds features
        'race_nige_horse_count', 'race_pace_cat', 'nige_candidate_count', 'senkou_ratio',
        'is_long_break', 'is_weight_changed_huge', 'is_class_up',
    ]
    
    feature_cols = [c for c in df.columns if c not in drop_cols and not c.endswith('_id')]
    X = df[feature_cols].copy()
    
    # Convert to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0).astype('float32')
    
    return X, feature_cols

def objective(trial, df, X, y, groups):
    """Optuna objective function"""
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.95),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.95),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-4, 1.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-4, 1.0, log=True),
        'random_state': 42,
        'verbose': -1
    }
    
    # 3-Fold CV by year
    cv_scores = []
    gkf = GroupKFold(n_splits=3)
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Get group sizes for LambdaRank
        train_races = df.iloc[train_idx]['race_id']
        val_races = df.iloc[val_idx]['race_id']
        train_groups = train_races.groupby(train_races).size().values
        val_groups = val_races.groupby(val_races).size().values
        
        train_data = lgb.Dataset(X_train, y_train, group=train_groups)
        val_data = lgb.Dataset(X_val, y_val, group=val_groups, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )
        
        # NDCG@1 is the key metric
        cv_scores.append(model.best_score['valid_0']['ndcg@1'])
    
    return np.mean(cv_scores)

def main():
    # Load data
    df = load_data()
    
    # Prepare features
    X, feature_cols = get_features(df)
    y = (df['rank'] == 1).astype(int)  # Target: is winner
    groups = df['date'].dt.year  # Group by year for CV
    
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Samples: {len(X)}")
    
    # Run optimization
    logger.info("Starting Optuna optimization...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, df, X, y, groups), n_trials=50, show_progress_bar=True)
    
    # Results
    logger.info("=" * 50)
    logger.info("Optimization Complete!")
    logger.info(f"Best NDCG@1: {study.best_value:.4f}")
    logger.info("Best Parameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    # Print Python dict format for easy copy
    print("\n# Best parameters (copy to PARAMS):")
    print("PARAMS = {")
    print("    'objective': 'lambdarank',")
    print("    'metric': 'ndcg',")
    print("    'ndcg_eval_at': [1, 3, 5],")
    print("    'boosting_type': 'gbdt',")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"    '{key}': {value:.6f},")
        else:
            print(f"    '{key}': {value},")
    print("    'random_state': 42,")
    print("    'verbose': -1")
    print("}")

if __name__ == "__main__":
    main()

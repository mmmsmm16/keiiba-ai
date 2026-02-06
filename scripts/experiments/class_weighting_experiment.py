"""
Class Weighting and Time-Weighted Sampling Experiment
======================================================
Tests:
1. Class weighting (scale_pos_weight)
2. Time-weighted sampling (exponential decay)

Usage:
  python scripts/experiments/class_weighting_experiment.py
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
TARGET_PATH = "data/temp_t2/T2_targets.parquet"
BASE_MODEL_PATH = "models/experiments/exp_t2_refined_v3/model.pkl"


def load_data():
    """Load and prepare data"""
    logger.info("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    targets = pd.read_parquet(TARGET_PATH)
    
    df['race_id'] = df['race_id'].astype(str)
    targets['race_id'] = targets['race_id'].astype(str)
    
    df = df.merge(targets[['race_id', 'horse_number', 'rank']], 
                  on=['race_id', 'horse_number'], how='left')
    df['date'] = pd.to_datetime(df['date'])
    df['is_win'] = (df['rank'] == 1).astype(int)
    
    return df


def run_experiment():
    """Run class weighting and time-weighted sampling experiment"""
    logger.info("=" * 60)
    logger.info("Class Weighting & Time-Weighted Sampling Experiment")
    logger.info("=" * 60)
    
    # Load data
    df = load_data()
    
    # Split by year
    df_train = df[df['date'].dt.year <= 2022].copy()
    df_valid = df[df['date'].dt.year == 2023].copy()
    df_test = df[df['date'].dt.year == 2024].copy()
    
    logger.info(f"Train (<=2022): {len(df_train)} records")
    logger.info(f"Valid (2023): {len(df_valid)} records")
    logger.info(f"Test (2024): {len(df_test)} records")
    
    # Load base model for feature names
    base_model = joblib.load(BASE_MODEL_PATH)
    base_features = base_model.feature_name()
    
    # Prepare features
    exclude_cols = ['race_id', 'horse_number', 'date', 'rank', 'is_win', 
                    'is_top2', 'is_top3', 'rank_str', 'year']
    feature_cols = [c for c in df_train.columns if c not in exclude_cols 
                    and c in base_features]
    
    X_train = df_train[feature_cols].copy()
    X_valid = df_valid[feature_cols].copy()
    X_test = df_test[feature_cols].copy()
    
    y_train = df_train['is_win']
    y_valid = df_valid['is_win']
    y_test = df_test['is_win']
    
    # Convert categorical
    for c in X_train.columns:
        if X_train[c].dtype == 'object' or X_train[c].dtype.name == 'category':
            X_train[c] = X_train[c].astype('category').cat.codes
            X_valid[c] = X_valid[c].astype('category').cat.codes
            X_test[c] = X_test[c].astype('category').cat.codes
        X_train[c] = X_train[c].fillna(-999)
        X_valid[c] = X_valid[c].fillna(-999)
        X_test[c] = X_test[c].fillna(-999)
    
    # Calculate class imbalance
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / pos_count
    logger.info(f"Class imbalance: {neg_count}:{pos_count} = {scale_pos_weight:.2f}")
    
    # Base params
    base_params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    results = []
    
    # ========================================
    # Experiment 1: Baseline
    # ========================================
    logger.info("\n--- Experiment 1: Baseline ---")
    
    train_data = lgb.Dataset(X_train.values, label=y_train)
    valid_data = lgb.Dataset(X_valid.values, label=y_valid, reference=train_data)
    
    model = lgb.train(
        base_params, train_data,
        num_boost_round=3000,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)]
    )
    
    pred_valid = model.predict(X_valid.values)
    pred_test = model.predict(X_test.values)
    
    auc_valid = roc_auc_score(y_valid, pred_valid)
    auc_test = roc_auc_score(y_test, pred_test)
    
    results.append({'name': 'Baseline', 'valid': auc_valid, 'test': auc_test})
    logger.info(f"Valid AUC: {auc_valid:.4f}, Test AUC: {auc_test:.4f}")
    
    # ========================================
    # Experiment 2: Class Weighting
    # ========================================
    logger.info("\n--- Experiment 2: Class Weighting ---")
    
    params_weighted = base_params.copy()
    params_weighted['scale_pos_weight'] = scale_pos_weight
    
    train_data = lgb.Dataset(X_train.values, label=y_train)
    valid_data = lgb.Dataset(X_valid.values, label=y_valid, reference=train_data)
    
    model = lgb.train(
        params_weighted, train_data,
        num_boost_round=3000,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)]
    )
    
    pred_valid = model.predict(X_valid.values)
    pred_test = model.predict(X_test.values)
    
    auc_valid = roc_auc_score(y_valid, pred_valid)
    auc_test = roc_auc_score(y_test, pred_test)
    
    results.append({'name': f'Class Weight ({scale_pos_weight:.1f})', 'valid': auc_valid, 'test': auc_test})
    logger.info(f"Valid AUC: {auc_valid:.4f}, Test AUC: {auc_test:.4f}")
    
    # ========================================
    # Experiment 3: Time-Weighted Sampling
    # ========================================
    logger.info("\n--- Experiment 3: Time-Weighted Sampling ---")
    
    # Calculate time weights (more recent = higher weight)
    max_year = df_train['date'].dt.year.max()
    years = df_train['date'].dt.year.values
    
    for decay_rate in [0.05, 0.10, 0.15]:
        weights = np.exp(-decay_rate * (max_year - years))
        
        train_data = lgb.Dataset(X_train.values, label=y_train, weight=weights)
        valid_data = lgb.Dataset(X_valid.values, label=y_valid, reference=train_data)
        
        model = lgb.train(
            base_params, train_data,
            num_boost_round=3000,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)]
        )
        
        pred_valid = model.predict(X_valid.values)
        pred_test = model.predict(X_test.values)
        
        auc_valid = roc_auc_score(y_valid, pred_valid)
        auc_test = roc_auc_score(y_test, pred_test)
        
        results.append({'name': f'Time Weight (decay={decay_rate})', 'valid': auc_valid, 'test': auc_test})
        logger.info(f"Decay {decay_rate}: Valid AUC: {auc_valid:.4f}, Test AUC: {auc_test:.4f}")
    
    # ========================================
    # Experiment 4: Combined (Class + Time)
    # ========================================
    logger.info("\n--- Experiment 4: Combined (Class + Time) ---")
    
    weights = np.exp(-0.10 * (max_year - years))  # Best decay from above
    
    params_weighted = base_params.copy()
    params_weighted['scale_pos_weight'] = scale_pos_weight
    
    train_data = lgb.Dataset(X_train.values, label=y_train, weight=weights)
    valid_data = lgb.Dataset(X_valid.values, label=y_valid, reference=train_data)
    
    model = lgb.train(
        params_weighted, train_data,
        num_boost_round=3000,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)]
    )
    
    pred_valid = model.predict(X_valid.values)
    pred_test = model.predict(X_test.values)
    
    auc_valid = roc_auc_score(y_valid, pred_valid)
    auc_test = roc_auc_score(y_test, pred_test)
    
    results.append({'name': 'Combined (Class + Time)', 'valid': auc_valid, 'test': auc_test})
    logger.info(f"Valid AUC: {auc_valid:.4f}, Test AUC: {auc_test:.4f}")
    
    # ========================================
    # Results Summary
    # ========================================
    print("\n" + "=" * 70)
    print(" Results Summary")
    print("=" * 70)
    print(f"\n{'Model':<35} | {'Valid AUC':<12} | {'Test AUC':<12}")
    print("-" * 65)
    
    baseline_test = results[0]['test']
    
    for r in results:
        diff = (r['test'] - baseline_test) * 100
        diff_str = f"({diff:+.2f}%)" if r['name'] != 'Baseline' else ""
        print(f"{r['name']:<35} | {r['valid']:<12.4f} | {r['test']:<12.4f} {diff_str}")
    
    # Best result
    best = max(results, key=lambda x: x['test'])
    print(f"\n🏆 Best: {best['name']} (Test AUC: {best['test']:.4f})")


if __name__ == "__main__":
    run_experiment()

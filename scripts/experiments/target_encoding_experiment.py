"""
Target Encoding Implementation
==============================
Implements smoothed target encoding for categorical features.
Uses K-Fold CV to prevent data leakage.

Usage:
  python scripts/experiments/target_encoding_experiment.py
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
TARGET_PATH = "data/temp_t2/T2_targets.parquet"
BASE_MODEL_PATH = "models/experiments/exp_t2_refined_v3/model.pkl"

# Categorical features to apply target encoding
TARGET_ENCODE_COLS = [
    'venue',           # 競馬場
    'grade_code',      # グレード
    'surface',         # 芝/ダート
    'going_code',      # 馬場状態
    'weather_code',    # 天候
    'sex',             # 性別
]


def smoothed_target_encode(train_df, valid_df, col, target, smooth=20):
    """
    Apply smoothed target encoding.
    
    Formula: (n_c * mean_c + m * global_mean) / (n_c + m)
    
    Args:
        train_df: Training dataframe
        valid_df: Validation dataframe
        col: Column to encode
        target: Target column name
        smooth: Smoothing parameter (higher = more regularization)
    
    Returns:
        encoded_train, encoded_valid
    """
    global_mean = train_df[target].mean()
    
    # Calculate per-category statistics
    agg = train_df.groupby(col)[target].agg(['mean', 'count'])
    
    # Apply smoothing
    smoothed_means = (agg['count'] * agg['mean'] + smooth * global_mean) / (agg['count'] + smooth)
    
    # Map to dataframes
    encoded_train = train_df[col].map(smoothed_means).fillna(global_mean)
    encoded_valid = valid_df[col].map(smoothed_means).fillna(global_mean)
    
    return encoded_train, encoded_valid


def kfold_target_encode(df, col, target, n_folds=5, smooth=20):
    """
    Apply target encoding using K-Fold to prevent leakage.
    
    For each fold, the encoding is calculated from other folds.
    """
    df = df.copy()
    encoded = pd.Series(index=df.index, dtype=float)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(df):
        train_fold = df.iloc[train_idx]
        val_fold = df.iloc[val_idx]
        
        global_mean = train_fold[target].mean()
        agg = train_fold.groupby(col)[target].agg(['mean', 'count'])
        smoothed_means = (agg['count'] * agg['mean'] + smooth * global_mean) / (agg['count'] + smooth)
        
        encoded.iloc[val_idx] = val_fold[col].map(smoothed_means).fillna(global_mean)
    
    return encoded


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
    """Run target encoding experiment"""
    logger.info("=" * 60)
    logger.info("Target Encoding Experiment")
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
    
    # Prepare baseline features
    exclude_cols = ['race_id', 'horse_number', 'date', 'rank', 'is_win', 
                    'is_top2', 'is_top3', 'rank_str', 'year']
    
    # ========================================
    # Baseline: Without target encoding
    # ========================================
    logger.info("\n--- Baseline Model (No Target Encoding) ---")
    
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
    
    # Train baseline
    params = {
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
    
    train_data = lgb.Dataset(X_train.values, label=y_train)
    valid_data = lgb.Dataset(X_valid.values, label=y_valid, reference=train_data)
    
    baseline_model = lgb.train(
        params, train_data,
        num_boost_round=3000,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)]
    )
    
    baseline_pred_valid = baseline_model.predict(X_valid.values)
    baseline_pred_test = baseline_model.predict(X_test.values)
    
    baseline_auc_valid = roc_auc_score(y_valid, baseline_pred_valid)
    baseline_auc_test = roc_auc_score(y_test, baseline_pred_test)
    
    logger.info(f"Baseline Valid AUC: {baseline_auc_valid:.4f}")
    logger.info(f"Baseline Test AUC: {baseline_auc_test:.4f}")
    
    # ========================================
    # With Target Encoding
    # ========================================
    logger.info("\n--- Model with Target Encoding ---")
    
    # Apply target encoding to training data
    df_train_te = df_train.copy()
    df_valid_te = df_valid.copy()
    df_test_te = df_test.copy()
    
    for col in TARGET_ENCODE_COLS:
        if col not in df_train_te.columns:
            logger.warning(f"Column {col} not found, skipping")
            continue
        
        # For training: use K-Fold target encoding
        df_train_te[f'{col}_te'] = kfold_target_encode(
            df_train_te, col, 'is_win', n_folds=5, smooth=20
        )
        
        # For valid/test: use training data statistics
        _, df_valid_te[f'{col}_te'] = smoothed_target_encode(
            df_train_te, df_valid_te, col, 'is_win', smooth=20
        )
        _, df_test_te[f'{col}_te'] = smoothed_target_encode(
            df_train_te, df_test_te, col, 'is_win', smooth=20
        )
        
        logger.info(f"Applied target encoding to: {col}")
    
    # Add target encoded features
    te_feature_cols = feature_cols + [f'{col}_te' for col in TARGET_ENCODE_COLS 
                                       if col in df_train.columns]
    
    X_train_te = df_train_te[te_feature_cols].copy()
    X_valid_te = df_valid_te[te_feature_cols].copy()
    X_test_te = df_test_te[te_feature_cols].copy()
    
    # Convert categorical
    for c in X_train_te.columns:
        if X_train_te[c].dtype == 'object' or X_train_te[c].dtype.name == 'category':
            X_train_te[c] = X_train_te[c].astype('category').cat.codes
            X_valid_te[c] = X_valid_te[c].astype('category').cat.codes
            X_test_te[c] = X_test_te[c].astype('category').cat.codes
        X_train_te[c] = X_train_te[c].fillna(-999)
        X_valid_te[c] = X_valid_te[c].fillna(-999)
        X_test_te[c] = X_test_te[c].fillna(-999)
    
    # Train with target encoding
    train_data_te = lgb.Dataset(X_train_te.values, label=y_train)
    valid_data_te = lgb.Dataset(X_valid_te.values, label=y_valid, reference=train_data_te)
    
    te_model = lgb.train(
        params, train_data_te,
        num_boost_round=3000,
        valid_sets=[valid_data_te],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)]
    )
    
    te_pred_valid = te_model.predict(X_valid_te.values)
    te_pred_test = te_model.predict(X_test_te.values)
    
    te_auc_valid = roc_auc_score(y_valid, te_pred_valid)
    te_auc_test = roc_auc_score(y_test, te_pred_test)
    
    logger.info(f"Target Encoding Valid AUC: {te_auc_valid:.4f}")
    logger.info(f"Target Encoding Test AUC: {te_auc_test:.4f}")
    
    # ========================================
    # Results Summary
    # ========================================
    print("\n" + "=" * 60)
    print(" Results Summary")
    print("=" * 60)
    print(f"\n{'Model':<25} | {'Valid AUC':<12} | {'Test AUC':<12}")
    print("-" * 55)
    print(f"{'Baseline':<25} | {baseline_auc_valid:<12.4f} | {baseline_auc_test:<12.4f}")
    print(f"{'+ Target Encoding':<25} | {te_auc_valid:<12.4f} | {te_auc_test:<12.4f}")
    
    valid_diff = (te_auc_valid - baseline_auc_valid) * 100
    test_diff = (te_auc_test - baseline_auc_test) * 100
    
    print(f"\n{'Improvement':<25} | {valid_diff:+.2f}%       | {test_diff:+.2f}%")
    
    if te_auc_test > baseline_auc_test:
        print("\n✅ Target Encoding improved test AUC!")
    else:
        print("\n⚠️ Target Encoding did not improve test AUC")


if __name__ == "__main__":
    run_experiment()

"""
Full Training Script with Optuna-Optimized Parameters
Uses the best parameters from Optuna HPO and excludes all leakage columns.

Usage: python scripts/run_train_optuna_best.py
"""
import os
import logging
import joblib
import yaml
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
OUTPUT_DIR = "models/experiments/optuna_best_full"

# Optuna Best Parameters (from 10-trial optimization)
BEST_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbosity': 1,
    'seed': 42,
    
    # Optimized hyperparameters
    'learning_rate': 0.046,
    'num_leaves': 61,
    'max_depth': 10,
    'min_data_in_leaf': 59,
    'feature_fraction': 0.57,
    'bagging_fraction': 0.89,
    'bagging_freq': 6,
    'lambda_l1': 2.85,
    'lambda_l2': 1.13,
}

# Leakage columns to exclude (identified during thorough analysis)
# These are ALL columns that contain information from the CURRENT race result
LEAKAGE_COLS = [
    # Direct race results
    'pass_1', 'pass_2', 'pass_3', 'pass_4', 'passing_rank',
    'last_3f', 'raw_time', 'time_diff', 'margin',
    'time',  # Current race finish time - MAJOR LEAK
    
    # Current race popularity/odds (final, not pre-race)
    'popularity',  # Final popularity - MAJOR LEAK
    'odds',  # Final odds
    'relative_popularity_rank',
    
    # Race-result derived metrics
    'slow_start_recovery', 'track_bias_disadvantage',
    'outer_frame_disadv',  # Derived from race results
    'wide_run',  # Derived from passing positions
    
    # Any other result-based columns
    'mean_time_diff_5',  # Contains current race time_diff in aggregation
    'horse_wide_run_rate',  # Contains current race wide_run
]


def load_data():
    """Load and prepare data with proper leakage exclusion"""
    logger.info("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    # Split: 2019-2022 train, 2023 valid, 2024+ test
    df_train = df[(df['year'] >= 2019) & (df['year'] <= 2022)]
    df_val = df[df['year'] == 2023]
    df_test = df[df['year'] >= 2024]
    
    logger.info(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    # Exclude meta, target and leakage columns
    meta_cols = [
        'race_id', 'horse_number', 'date', 'rank', 'odds_final', 
        'is_win', 'is_top2', 'is_top3', 'year', 'rank_str'
    ]
    id_cols = ['horse_id', 'mare_id', 'sire_id', 'jockey_id', 'trainer_id']
    
    exclude_all = meta_cols + LEAKAGE_COLS + id_cols
    feature_cols = [c for c in df.columns if c not in exclude_all]
    
    logger.info(f"Features: {len(feature_cols)} columns")
    
    # Extract features and targets
    X_train = df_train[feature_cols].copy()
    y_train = (df_train['rank'] == 1).astype(int)
    X_val = df_val[feature_cols].copy()
    y_val = (df_val['rank'] == 1).astype(int)
    X_test = df_test[feature_cols].copy() if len(df_test) > 0 else None
    y_test = (df_test['rank'] == 1).astype(int) if len(df_test) > 0 else None
    
    # Handle non-numeric dtypes
    for col in feature_cols:
        if X_train[col].dtype.name == 'category' or X_train[col].dtype == 'object':
            X_train[col] = X_train[col].astype('category').cat.codes
            X_val[col] = X_val[col].astype('category').cat.codes
            if X_test is not None:
                X_test[col] = X_test[col].astype('category').cat.codes
        else:
            X_train[col] = X_train[col].fillna(-999)
            X_val[col] = X_val[col].fillna(-999)
            if X_test is not None:
                X_test[col] = X_test[col].fillna(-999)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, df_val, df_test


def main():
    logger.info("=" * 60)
    logger.info("Full Training with Optuna-Optimized Parameters")
    logger.info("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, df_val, df_test = load_data()
    
    # Create datasets
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    # Train
    logger.info("Training with optimized parameters...")
    callbacks = [
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100)
    ]
    
    model = lgb.train(
        BEST_PARAMS,
        lgb_train,
        num_boost_round=3000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )
    
    # Evaluate
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    
    # Validation
    val_preds = model.predict(X_val)
    val_auc = roc_auc_score(y_val, val_preds)
    val_logloss = log_loss(y_val, val_preds)
    logger.info(f"Validation (2023): AUC={val_auc:.5f}, LogLoss={val_logloss:.5f}")
    
    # Test
    if X_test is not None and len(X_test) > 0:
        test_preds = model.predict(X_test)
        test_auc = roc_auc_score(y_test, test_preds)
        test_logloss = log_loss(y_test, test_preds)
        logger.info(f"Test (2024+):      AUC={test_auc:.5f}, LogLoss={test_logloss:.5f}")
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, "model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"\nModel saved to: {model_path}")
    
    # Feature Importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    importance.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    
    print("\nTop 20 Important Features:")
    for i, row in importance.head(20).iterrows():
        print(f"  {row['feature']:<35}: {row['importance']:.2f}")
    
    # Save metrics
    metrics = {
        'validation_auc': val_auc,
        'validation_logloss': val_logloss,
        'test_auc': test_auc if X_test is not None else None,
        'test_logloss': test_logloss if X_test is not None else None,
        'n_features': len(feature_cols),
        'best_iteration': model.best_iteration,
        'params': BEST_PARAMS
    }
    
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save feature list
    with open(os.path.join(OUTPUT_DIR, "feature_list.txt"), 'w') as f:
        f.write('\n'.join(feature_cols))
    
    logger.info(f"\nAll artifacts saved to: {OUTPUT_DIR}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

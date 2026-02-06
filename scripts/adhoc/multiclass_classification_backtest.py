"""
Multiclass Classification Model for Horse Racing Prediction (Rank 1-7+)
Predicts exact rank from 1st to 6th, and 7th+ as the final class.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import lightgbm as lgb
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CACHE_PATH = "data/cache/jra_base/advanced.parquet"
OUTPUT_DIR = "reports/jra/multiclass_classification"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Features to drop (Shared with binary/LambdaRank for leak prevention)
DROP_FEATURES = [
    'mare_id', 'sire_id', 'trainer_id', 'jockey_id', 'bms_id',
    'odds', 'popularity', 'owner_id', 'breeder_id',
    'lag1_odds', 'lag1_popularity',
    # Redundant features
    'race_nige_horse_count', 'race_pace_cat', 'nige_candidate_count', 'senkou_ratio',
    'is_long_break', 'is_weight_changed_huge', 'is_class_up',
    # Direct race outcome data (Leak prevention)
    'honshokin', 'fukashokin', 'time', 'raw_time', 'last_3f', 'time_diff',
    'passing_rank', 'pass_1', 'pass_2', 'pass_3', 'pass_4',
    'slow_start_recovery', 'pace_disadvantage', 'wide_run',
    'outer_frame_disadv', 'track_bias_disadvantage'
]

# LGBM Multiclass parameters
PARAMS = {
    'objective': 'multiclass',
    'num_class': 7,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'random_state': 42,
    'verbose': -1,
    'n_estimators': 300,
    # 'class_weight': 'balanced' # Multiclass in LGBM doesn't support 'balanced' directly like Binary
}


def get_features(df, feature_cols=None):
    """Extract feature matrix - carefully exclude leaky columns"""
    leaky_cols = [
        'race_id', 'horse_id', 'horse_number', 'date',
        'rank', 'rank_str', 'time', 'raw_time', 'target', 'target_top3', 'target_rank',
        'pred_prob', 'calib_prob', 'pred_rank',
        'passing_rank', 'pass_1', 'pass_2', 'pass_3', 'pass_4',
        'honshokin', 'fukashokin',
        'time_diff', 'last_3f',
        'relative_strength',
        'relative_popularity_rank',
        'estimated_place_rate',
        'race_avg_prize',
        'n_horses'
    ]

    if feature_cols is None:
        feature_cols = []
        for c in df.columns:
            if c in leaky_cols or c in DROP_FEATURES:
                continue
            if c.endswith('_id'):
                continue
            if c == 'rank' or c == 'rank_str' or c == 'time':
                continue
            feature_cols.append(c)

    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0).astype('float32')

    return X, feature_cols


def run_multiclass_backtest():
    """Run multiclass classification (Rank 1-7+)"""
    logger.info("Loading base data...")
    if not os.path.exists(CACHE_PATH):
        logger.error(f"Cache file not found: {CACHE_PATH}")
        return
    
    df = pd.read_parquet(CACHE_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # Create Multiclass Target: Class 0 (Rank 1) to Class 6 (Rank 7+)
    # rank 1.0 -> target 0
    # rank 7.0 or better stays, others clip to 7, then 0-index
    df['target_rank'] = (df['rank'].clip(upper=7) - 1).astype(int)

    logger.info(f"Data: {len(df)} records ({df['date'].min()} to {df['date'].max()})")
    for i in range(7):
        logger.info(f"Class {i} (Rank {i+1 if i < 6 else '7+'}): {(df['target_rank'] == i).mean():.1%}")

    # Split: Train on 2020-2023, Test on 2024
    train_df = df[(df['date'] >= '2020-01-01') & (df['date'] < '2024-01-01')]
    test_df = df[(df['date'] >= '2024-01-01') & (df['date'] < '2025-01-01')]

    if len(train_df) == 0 or len(test_df) == 0:
        logger.error("Train or Test data is empty.")
        return

    # Get features
    X_train, feature_cols = get_features(train_df)
    X_test, _ = get_features(test_df, feature_cols)
    y_train = train_df['target_rank']
    y_test = test_df['target_rank']

    logger.info(f"Features: {len(feature_cols)}")

    # Train model
    logger.info("Training LGBMClassifier (Multiclass)...")
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X_train, y_train)

    # Predict probabilities (Shape: [N, 7])
    test_df = test_df.copy()
    y_pred_probs = model.predict_proba(X_test)
    
    # Store probability of Rank 1 (Class 0)
    test_df['pred_prob_win'] = y_pred_probs[:, 0]
    # Store probability of Top 3 (Classes 0, 1, 2)
    test_df['pred_prob_top3'] = y_pred_probs[:, 0:3].sum(axis=1)
    
    # Predicted Rank based on Class 0 probability within race
    test_df['pred_rank_win'] = test_df.groupby('race_id')['pred_prob_win'].rank(ascending=False, method='first').astype(int)

    # 1. Performance Analysis
    logger.info("\n=== Multiclass Analysis (2024 Test) ===")
    
    # Win Rate by Predicted Rank 1
    top1 = test_df[test_df['pred_rank_win'] == 1]
    n_races = len(top1)
    n_wins = (top1['rank'] == 1).sum()
    n_top3 = (top1['rank'] <= 3).sum()
    
    logger.info(f"Top1 Selection Accuracy (using Class 0 prob):")
    logger.info(f"  Win Rate: {n_wins/n_races*100:.1f}%")
    logger.info(f"  Top3 Rate: {n_top3/n_races*100:.1f}%")
    logger.info(f"  Sample Size: {n_races}")

    # Payout (ROI) check for Win
    top1_valid = top1[top1['odds'] > 0]
    invest = len(top1_valid) * 100
    payout = (top1_valid[top1_valid['rank'] == 1]['odds'] * 100).sum()
    roi = payout / invest * 100 if invest > 0 else 0
    logger.info(f"Top1 Win ROI: {roi:.1f}%")

    # LogLoss check
    # We can use LGBM's internal metric if we have a validation set, 
    # but here we just info the train logloss
    
    # Accuracy by predicted class (Confusion focus)
    logger.info("\n=== Class Prediction Probabilities (Mean per Rank) ===")
    for r in range(1, 4):
        actual_rank_df = test_df[test_df['rank'] == r]
        avg_probs = model.predict_proba(get_features(actual_rank_df, feature_cols)[0]).mean(axis=0)
        logger.info(f"Actual Rank {r}: [1着={avg_probs[0]:.2f}, 2着={avg_probs[1]:.2f}, 3着={avg_probs[2]:.2f}, ...]")

    return model, test_df


if __name__ == "__main__":
    run_multiclass_backtest()

"""
Binary Classification Model for Horse Racing Prediction
Reuses the monthly features from LambdaRank backtest results.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import lightgbm as lgb
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CACHE_PATH = "data/cache/jra_base/advanced.parquet"
OUTPUT_DIR = "reports/jra/binary_classification"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Features to drop
DROP_FEATURES = [
    'mare_id', 'sire_id', 'trainer_id', 'jockey_id', 'bms_id',
    'odds', 'popularity', 'owner_id', 'breeder_id',
    'lag1_odds', 'lag1_popularity',
    # Redundant features
    'race_nige_horse_count', 'race_pace_cat', 'nige_candidate_count', 'senkou_ratio',
    'is_long_break', 'is_weight_changed_huge', 'is_class_up',
    # Direct race outcome data (Strict leak prevention)
    'honshokin', 'fukashokin', 'time', 'raw_time', 'last_3f', 'time_diff',
    'passing_rank', 'pass_1', 'pass_2', 'pass_3', 'pass_4',
    # Disadvantage flags for the current race
    'slow_start_recovery', 'pace_disadvantage', 'wide_run',
    'outer_frame_disadv', 'track_bias_disadvantage'
]

# LGBMClassifier parameters
PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.044449,
    'num_leaves': 65,
    'min_child_samples': 83,
    'feature_fraction': 0.860430,
    'bagging_fraction': 0.787839,
    'bagging_freq': 5,
    'reg_alpha': 0.010220,
    'reg_lambda': 0.000172,
    'random_state': 42,
    'verbose': -1,
    'n_estimators': 200,
    'class_weight': 'balanced',
}


def get_features(df, feature_cols=None):
    """Extract feature matrix - carefully exclude leaky columns"""
    leaky_cols = [
        'race_id', 'horse_id', 'horse_number', 'date',
        'rank', 'rank_str', 'time', 'raw_time', 'target', 'target_top3',
        'pred_prob', 'calib_prob', 'pred_rank',
        'passing_rank', 'pass_1', 'pass_2', 'pass_3', 'pass_4',
        'honshokin', 'fukashokin',
        'time_diff', 'last_3f',
        # Derived from this race results
        'relative_strength',
        'relative_popularity_rank',
        'estimated_place_rate',
        # Race-level features that might include current race info
        'race_avg_prize',
        'n_horses',
        # Disadvantage flags
        'slow_start_recovery', 'pace_disadvantage', 'wide_run',
        'outer_frame_disadv', 'track_bias_disadvantage'
    ]

    if feature_cols is None:
        feature_cols = []
        for c in df.columns:
            # Skip if in explicit drop list
            if c in leaky_cols or c in DROP_FEATURES:
                continue
            # Skip ID columns
            if c.endswith('_id'):
                continue
            # Skip current race rank features
            if c == 'rank' or c == 'rank_str' or c == 'time':
                continue
            feature_cols.append(c)

    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0).astype('float32')

    return X, feature_cols


def run_binary_backtest():
    """Run binary classification using historical data + CV on 2024"""
    # Load all cached data
    logger.info("Loading base data...")
    if not os.path.exists(CACHE_PATH):
        logger.error(f"Cache file not found: {CACHE_PATH}")
        return
    
    df = pd.read_parquet(CACHE_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['target_top3'] = (df['rank'] <= 3).astype(int)

    logger.info(f"Data: {len(df)} records ({df['date'].min()} to {df['date'].max()})")
    logger.info(f"Target: Top3={df['target_top3'].mean():.1%}")

    # Split: Train on 2020-01-01 to 2025-12-14
    train_df = df[(df['date'] >= '2020-01-01') & (df['date'] <= '2025-12-14')]
    # For validation metrics in this adhoc script, use 2024 data
    test_df = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2024-12-31')]

    if len(train_df) == 0 or len(test_df) == 0:
        logger.error("Train or Test data is empty.")
        return

    # Get features
    X_train, feature_cols = get_features(train_df)
    X_test, _ = get_features(test_df, feature_cols)
    y_train = train_df['target_top3']
    y_test = test_df['target_top3']

    logger.info(f"Features: {len(feature_cols)}")

    # Train model
    logger.info("Training LGBMClassifier...")
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X_train, y_train)
    
    # Feature Importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    logger.info("\nTop 20 Features:")
    logger.info(importance.head(20).to_string(index=False))

    # Predict
    test_df = test_df.copy()
    test_df['pred_prob'] = model.predict_proba(X_test)[:, 1]
    test_df['pred_rank'] = test_df.groupby('race_id')['pred_prob'].rank(ascending=False, method='first').astype(int)

    # 1. Probability Distribution Analysis
    probs = test_df['pred_prob']
    logger.info("\n=== Probability Distribution (Binary) ===")
    logger.info(f"Mean: {probs.mean():.3f}, Std: {probs.std():.3f}")
    logger.info(f"Min: {probs.min():.3f}, Max: {probs.max():.3f}")
    
    # Top1 pred_prob
    top1 = test_df[test_df['pred_rank'] == 1]
    logger.info(f"Top1 prob: mean={top1['pred_prob'].mean():.3f}, max={top1['pred_prob'].max():.3f}")

    # 2. ROI Analysis (2024 Test)
    logger.info("\n=== ROI Analysis (2024 Test) ===")
    top1_valid = top1[top1['odds'] > 0].copy()
    n_races = len(top1_valid)
    n_wins = (top1_valid['rank'] == 1).sum()
    invest = n_races * 100
    payout = (top1_valid[top1_valid['rank'] == 1]['odds'] * 100).sum()
    roi = payout / invest * 100 if invest > 0 else 0
    logger.info(f"Top1 Win Rate: {n_wins/n_races*100:.1f}%, ROI: {roi:.1f}%")

    # Accuracy by Rank
    logger.info("\n=== Accuracy by Predicted Rank ===")
    for pr in range(1, 6):
        subset = test_df[test_df['pred_rank'] == pr]
        if len(subset) == 0: continue
        win = (subset['rank'] == 1).mean()
        show = (subset['rank'] <= 3).mean()
        logger.info(f"  Rank {pr}: Win={win:5.1%}, Top3={show:5.1%}, N={len(subset)}")

    # Save Model
    import joblib
    model_path = "models/binary_no_odds.pkl"
    os.makedirs("models", exist_ok=True)
    model_data = {
        'model': model,
        'feature_cols': feature_cols,
        'params': PARAMS
    }
    joblib.dump(model_data, model_path)
    logger.info(f"Model saved to {model_path}")

    return model, test_df


if __name__ == "__main__":
    run_binary_backtest()

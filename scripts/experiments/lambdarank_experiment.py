"""
LambdaRank Experiment Script
============================
Trains a LightGBM LambdaRank model optimized for NDCG.
Requires query data (grouping by race_id) for ranking.

Usage:
  python scripts/experiments/lambdarank_experiment.py
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.metrics import ndcg_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
TARGET_PATH = "data/temp_t2/T2_targets.parquet"
BASE_MODEL_PATH = "models/experiments/exp_t2_refined_v3/model.pkl"
OUTPUT_MODEL_PATH = "models/experiments/exp_lambdarank/model.pkl"

def load_data():
    """Load and prepare data for ranking"""
    logger.info("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    # Merge targets if rank is missing
    if 'rank' not in df.columns:
        logger.info("Rank not found, merging from T2_targets...")
        targets = pd.read_parquet(TARGET_PATH)
        df['race_id'] = df['race_id'].astype(str)
        targets['race_id'] = targets['race_id'].astype(str)
        df = df.merge(targets[['race_id', 'horse_number', 'rank']], 
                      on=['race_id', 'horse_number'], how='left')
    else:
        logger.info("Rank column exists in dataset. Skipping target merge.")
        
    df['race_id'] = df['race_id'].astype(str)
    df['date'] = pd.to_datetime(df['date'])
    
    # Create relevance score (relevance = 1/rank, or binary)
    # For LambdaRank, higher label is better.
    # Rank 1 -> 3, Rank 2 -> 2, Rank 3 -> 1, Others -> 0
    df['relevance'] = 0
    df.loc[df['rank'] == 1, 'relevance'] = 3
    df.loc[df['rank'] == 2, 'relevance'] = 2
    df.loc[df['rank'] == 3, 'relevance'] = 1
    
    return df

def prepare_lgb_dataset(df, feature_cols):
    """
    Prepare LightGBM dataset with groups (queries)
    Data must be sorted by group_id (race_id)
    """
    # Sort by race_id needed for group boundaries
    df = df.sort_values('race_id')
    
    X = df[feature_cols].copy()
    y = df['relevance'].values
    group = df.groupby('race_id').size().values
    
    # Convert categorical
    for c in X.columns:
        if X[c].dtype == 'object' or X[c].dtype.name == 'category':
            X[c] = X[c].astype('category').cat.codes
        X[c] = X[c].fillna(-999)
    
    ds = lgb.Dataset(X, label=y, group=group)
    return ds, X, y, df['race_id'].values

def evaluate_ndcg(model, X, y, race_ids, k=3):
    """Calculate average NDCG@k"""
    logger.info(f"Evaluating NDCG@{k}...")
    
    preds = model.predict(X)
    
    # Create DF for calc
    df_eval = pd.DataFrame({
        'race_id': race_ids,
        'relevance': y,
        'score': preds
    })
    
    ndcg_list = []
    for rid, grp in df_eval.groupby('race_id'):
        if len(grp) < 2:
            continue
        # ndcg_score requires shape (n_samples, n_items)
        true_relevance = [grp['relevance'].values]
        scores = [grp['score'].values]
        try:
            score = ndcg_score(true_relevance, scores, k=k)
            ndcg_list.append(score)
        except Exception:
            pass
            
    return np.mean(ndcg_list)

def run_experiment():
    logger.info("=" * 60)
    logger.info("LambdaRank Experiment")
    logger.info("=" * 60)
    
    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    
    df = load_data()
    
    # Load feature names from base model
    # Feature Selection
    meta_cols = ['race_id', 'horse_number', 'date', 'rank', 'relevance', 'rank_str', 'is_win', 'is_top2', 'is_top3', 'year']
    leakage_cols = [
        'pass_1', 'pass_2', 'pass_3', 'pass_4', 'passing_rank',
        'last_3f', 'raw_time', 'time_diff', 'margin', 'time',
        'popularity', 'odds', 'odds_final', 'relative_popularity_rank',
        'slow_start_recovery', 'track_bias_disadvantage',
        'outer_frame_disadv', 'wide_run', 'mean_time_diff_5', 'horse_wide_run_rate'
    ]
    id_cols = ['horse_id', 'mare_id', 'sire_id', 'jockey_id', 'trainer_id']
    exclude_all = set(meta_cols + leakage_cols + id_cols)
    
    feature_cols = [c for c in df.columns if c not in exclude_all]
    logger.info(f"Selected {len(feature_cols)} features for training.")
    
    # Split
    df_train = df[df['date'].dt.year <= 2022].copy()
    df_valid = df[df['date'].dt.year == 2023].copy()
    df_test = df[df['date'].dt.year == 2024].copy()
    
    logger.info(f"Train: {len(df_train)}")
    logger.info(f"Valid: {len(df_valid)}")
    logger.info(f"Test: {len(df_test)}")
    
    # Prepare datasets
    train_ds, _, _, _ = prepare_lgb_dataset(df_train, feature_cols)
    valid_ds, X_valid, y_valid, valid_rids = prepare_lgb_dataset(df_valid, feature_cols)
    _, X_test, y_test, test_rids = prepare_lgb_dataset(df_test, feature_cols)
    
    # Parameters for LambdaRank
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
        'label_gain': [0, 1, 3, 7] # Gains for relevance 0, 1, 2, 3
    }
    
    logger.info("Training LambdaRank model...")
    model = lgb.train(
        params,
        train_ds,
        num_boost_round=5000,
        valid_sets=[valid_ds],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)]
    )
    
    # Evaluate
    logger.info("\nEvaluation Results:")
    
    valid_ndcg = evaluate_ndcg(model, X_valid, y_valid, valid_rids, k=3)
    test_ndcg = evaluate_ndcg(model, X_test, y_test, test_rids, k=3)
    
    print(f"\nValid NDCG@3: {valid_ndcg:.4f}")
    print(f"Test NDCG@3:  {test_ndcg:.4f}")
    
    # Save model
    joblib.dump(model, OUTPUT_MODEL_PATH)
    logger.info(f"Saved model to {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    run_experiment()

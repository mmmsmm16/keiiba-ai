"""
Evaluation Model Training & Backtest Script
Objective: Train on 2020-2024 and evaluate on 2025 data (Walk-forward validation).
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
EVAL_MODEL_DIR = "models/eval"
os.makedirs(EVAL_MODEL_DIR, exist_ok=True)

PARAMS_BIN = {
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

PARAMS_RANK = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 3, 5],
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'random_state': 42,
    'verbose': -1,
    'n_estimators': 200,
}

def get_features(df, feature_cols=None):
    drop_list = [
        'mare_id', 'sire_id', 'trainer_id', 'jockey_id', 'bms_id',
        'odds', 'popularity', 'owner_id', 'breeder_id', 'rank_str',
        'lag1_odds', 'lag1_popularity',
        'honshokin', 'fukashokin', 'time', 'raw_time', 'last_3f', 'time_diff',
        'passing_rank', 'pass_1', 'pass_2', 'pass_3', 'pass_4',
        'relative_popularity_rank',
        # Importance 0 Features (v18 Pruning) - Keeping these as they were safe
        'lag1_last_3f_is_missing', 'momentum_slope', 'surface', 'weather', 'state',
        'mean_rank_all_is_missing', 'mean_time_diff_5_is_missing', 'title',
        'race_opponent_strength_is_missing', 'lag1_time_diff_is_missing',
        'mean_last_3f_5_is_missing', 'weight_diff_sign', 'abnormal_code', 'horse_name',
        'distance_category', 'sex', 'mean_rank_norm_5_is_missing', 'distance_type',
        'lag1_rank_is_missing', 'frame_zone', 'direction', 'first_distance_cat',
        'weight_is_missing', 'year', 'lag1_rank_norm_is_missing'
    ]
    leaky_cols = ['race_id', 'horse_id', 'horse_number', 'date', 'rank', 'target', 'target_top3', 'rank_str', 'relevance']
    
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in leaky_cols and c not in drop_list and not c.endswith('_id')]
        # Reverted strict duplicate cleaning as it harmed performance (v19 regression)
        # feature_cols = sorted(list(set(feature_cols)))

    X = df[feature_cols].copy()
    
    # Reverted duplicate column removal in X
    # if X.columns.duplicated().any():
    #    X = X.loc[:, ~X.columns.duplicated()]

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0).astype('float32')
    return X, feature_cols

def calculate_metrics(df, pred_col, label="Model"):
    # Calculate within-race rank
    df_eval = df.copy()
    df_eval['pred_rank'] = df_eval.groupby('race_id')[pred_col].rank(ascending=False, method='first').astype(int)
    
    top1 = df_eval[df_eval['pred_rank'] == 1]
    top1_valid = top1[top1['odds'] > 0]
    n_races = len(top1_valid)
    if n_races == 0: return None
    
    n_wins = (top1_valid['rank'] == 1).sum()
    win_rate = n_wins / n_races * 100
    top3_rate = (top1['rank'] <= 3).mean() * 100
    
    invest = n_races * 100
    payout = (top1_valid[top1_valid['rank'] == 1]['odds'] * 100).sum()
    roi = payout / invest * 100
    
    return {
        'label': label,
        'win_rate': win_rate,
        'top3_rate': top3_rate,
        'roi': roi,
        'n_races': n_races
    }

def run_evaluation():
    logger.info("Loading data for evaluation...")
    df = pd.read_parquet(CACHE_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['target_top3'] = (df['rank'] <= 3).astype(int)
    # Relevance label for ranking: 1st=3, 2nd=2, 3rd=1, others=0
    df['relevance'] = df['rank'].apply(lambda x: max(0, 4 - x) if x > 0 else 0)
    
    # Validation Split
    train_df = df[(df['date'] >= '2020-01-01') & (df['date'] <= '2024-12-31')].sort_values(['date', 'race_id'])
    test_df = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-12-14')].sort_values(['date', 'race_id'])
    
    if len(test_df) == 0:
        logger.error("No 2025 data found for evaluation!")
        return

    logger.info(f"Training on {len(train_df)} rows up to 2024. Evaluating on {len(test_df)} rows in 2025.")
    
    X_train, feature_cols = get_features(train_df)
    X_test, _ = get_features(test_df, feature_cols)
    
    # 1. Binary Model
    logger.info("Training Binary evaluation model...")
    y_train_bin = train_df['target_top3']
    model_bin = lgb.LGBMClassifier(**PARAMS_BIN)
    model_bin.fit(X_train, y_train_bin)
    
    # 2. Ranking Model
    logger.info("Training Ranker evaluation model...")
    y_train_rank = train_df['relevance']
    train_groups = train_df.groupby('race_id').size().values
    model_rank = lgb.LGBMRanker(**PARAMS_RANK)
    model_rank.fit(X_train, y_train_rank, group=train_groups)

    # Output Binary Feature Importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model_bin.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\n=== TOP FEATURES (Binary Model) ===")
    print(importance.head(20))

    # Predict
    test_df = test_df.copy()
    test_df['score_bin'] = model_bin.predict_proba(X_test)[:, 1]
    test_df['score_rank'] = model_rank.predict(X_test)
    
    # Normalize scores for ensemble (Min-Max within race)
    def normalize(s):
        if s.max() == s.min(): return s * 0 + 0.5
        return (s - s.min()) / (s.max() - s.min())
    
    test_df['score_bin_norm'] = test_df.groupby('race_id')['score_bin'].transform(normalize)
    test_df['score_rank_norm'] = test_df.groupby('race_id')['score_rank'].transform(normalize)
    
    # Ensemble: Weighted Average (0.6 Binary, 0.4 Ranker)
    test_df['score_ensemble'] = test_df['score_bin_norm'] * 0.6 + test_df['score_rank_norm'] * 0.4
    
    # Calculate Metrics
    results = []
    results.append(calculate_metrics(test_df, 'score_bin', "Binary Only"))
    results.append(calculate_metrics(test_df, 'score_rank', "Ranker Only"))
    results.append(calculate_metrics(test_df, 'score_ensemble', "Ensemble (0.6/0.4)"))
    
    print("\n=== EVALUATION RESULTS (Test Period: 2025) ===")
    print(f"{'Model':<20} | {'WinRate':>8} | {'Top3Rate':>10} | {'ROI':>8} | {'Races':>6}")
    print("-" * 65)
    for res in results:
        if res:
            print(f"{res['label']:<20} | {res['win_rate']:7.1f}% | {res['top3_rate']:9.1f}% | {res['roi']:7.1f}% | {res['n_races']:6}")
    
    # Decile Analysis for Ensemble
    print("\n=== Calibration Analysis (Ensemble, 2025) ===")
    test_df['decile'] = pd.qcut(test_df['score_ensemble'], 10, labels=False, duplicates='drop')
    decile_stats = test_df.groupby('decile').agg({
        'score_ensemble': 'mean',
        'target_top3': ['mean', 'count']
    })
    print(f"{'Bin':>5} | {'Ens Score':>10} | {'Actual Rate':>10} | {'Count':>7}")
    print("-" * 45)
    for decile, row in decile_stats.iterrows():
        print(f"{decile:>5} | {row[('score_ensemble', 'mean')]:10.1%} | {row[('target_top3', 'mean')]:10.1%} | {int(row[('target_top3', 'count')]):>7}")

    # Save Models
    joblib.dump({'model': model_bin, 'feature_cols': feature_cols}, f"{EVAL_MODEL_DIR}/binary_eval_v18.pkl")
    joblib.dump({'model': model_rank, 'feature_cols': feature_cols}, f"{EVAL_MODEL_DIR}/ranker_eval_v18.pkl")
    logger.info("Evaluation complete.")

if __name__ == "__main__":
    run_evaluation()

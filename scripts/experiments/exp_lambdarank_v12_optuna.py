"""
LambdaRank Experiment with Optuna Hyperparameter Optimization (Batch 4)
========================================================================
Uses Optuna to find optimal LightGBM LambdaRank hyperparameters,
then trains final model with best params.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import GroupKFold
import joblib

sys.path.append('/workspace')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/preprocessed_data_v12.parquet"
OUTPUT_DIR = "models/experiments/exp_lambdarank_v12_batch4_optuna"
N_TRIALS = 30  # Optuna trials

os.makedirs(OUTPUT_DIR, exist_ok=True)


def ndcg_at_k(y_true, y_pred, groups, k=3):
    """Calculate NDCG@k for ranking"""
    from sklearn.metrics import ndcg_score
    scores = []
    idx = 0
    for g in groups:
        y_t = y_true[idx:idx+g]
        y_p = y_pred[idx:idx+g]
        if len(y_t) >= k and y_t.sum() > 0:
            s = ndcg_score([y_t], [y_p], k=k)
            scores.append(s)
        idx += g
    return np.mean(scores) if scores else 0.0


def load_data():
    """Load and prepare data"""
    logger.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    # Convert relevance
    df['relevance'] = 0
    df.loc[df['rank'] == 1, 'relevance'] = 3
    df.loc[df['rank'] == 2, 'relevance'] = 2
    df.loc[df['rank'] == 3, 'relevance'] = 1
    df['is_win'] = (df['rank'] == 1).astype(int)
    
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    # Split
    train_df = df[(df['year'] >= 2019) & (df['year'] <= 2022)]
    valid_df = df[df['year'] == 2023]
    test_df = df[df['year'] == 2024]
    
    # Feature selection
    meta_cols = ['race_id', 'horse_number', 'date', 'rank', 'relevance', 'rank_str', 'is_win', 'is_top2', 'is_top3', 'year']
    leakage_cols = [
        'pass_1', 'pass_2', 'pass_3', 'pass_4', 'passing_rank',
        'last_3f', 'raw_time', 'time_diff', 'margin', 'time',
        'popularity', 'odds', 'odds_final', 'relative_popularity_rank',
        'slow_start_recovery', 'track_bias_disadvantage',
        'outer_frame_disadv', 'wide_run', 'mean_time_diff_5', 'horse_wide_run_rate',
        'yoso_time_diff', 'yoso_rank_diff',
        'odds_10min', 'odds_60min', 'odds_ratio_60_10', 'odds_ratio_10min',
        'odds_log_ratio_10min', 'odds_diff_10min', 'odds_diff_60min',
        'std_odds', 'min_odds', 'max_odds', 'range_odds', 'volatility_odds',
        'rank_diff_10min', 'rank_diff_60min', 'trend_odds',
        'jockey_age', 'jockey_career_years', 'jockey_belong_code',
        'trainer_age', 'trainer_career_years', 'trainer_belong_code',
        # Corner features that use race result (leakage)
        'corner_position_change', 'makuri_positions', 'late_charge'
    ]
    id_cols = ['horse_id', 'mare_id', 'sire_id', 'jockey_id', 'trainer_id']
    exclude_all = set(meta_cols + leakage_cols + id_cols)
    
    # Exclude string/object columns
    string_cols = [c for c in df.columns if df[c].dtype == 'object']
    exclude_all.update(string_cols)
    
    feature_cols = [c for c in df.columns if c not in exclude_all]
    logger.info(f"Selected {len(feature_cols)} features for training.")
    
    return train_df, valid_df, test_df, feature_cols


def create_lgb_dataset(df, feature_cols):
    """Create LightGBM dataset for ranking"""
    X = df[feature_cols].values.astype(np.float32)
    y = df['relevance'].values
    groups = df.groupby('race_id').size().values
    return X, y, groups


def objective(trial, X_train, y_train, groups_train, X_valid, y_valid, groups_valid):
    """Optuna objective function"""
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, group=groups_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, group=groups_valid, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    y_pred = model.predict(X_valid)
    ndcg = ndcg_at_k(y_valid, y_pred, groups_valid, k=3)
    
    return ndcg


def main():
    logger.info("=" * 60)
    logger.info("LambdaRank + Optuna HPO (Batch 4)")
    logger.info("=" * 60)
    
    train_df, valid_df, test_df, feature_cols = load_data()
    
    X_train, y_train, groups_train = create_lgb_dataset(train_df, feature_cols)
    X_valid, y_valid, groups_valid = create_lgb_dataset(valid_df, feature_cols)
    X_test, y_test, groups_test = create_lgb_dataset(test_df, feature_cols)
    
    logger.info(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    
    # Optuna optimization
    logger.info(f"\nStarting Optuna optimization ({N_TRIALS} trials)...")
    study = optuna.create_study(direction='maximize', study_name='lambdarank_hpo')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, groups_train, X_valid, y_valid, groups_valid),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )
    
    logger.info(f"\nBest trial: {study.best_trial.number}")
    logger.info(f"Best NDCG@3: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    # Save best params
    best_params = study.best_params.copy()
    best_params['objective'] = 'lambdarank'
    best_params['metric'] = 'ndcg'
    best_params['ndcg_eval_at'] = [1, 3, 5]
    best_params['boosting_type'] = 'gbdt'
    best_params['verbosity'] = -1
    best_params['seed'] = 42
    
    pd.Series(best_params).to_json(os.path.join(OUTPUT_DIR, "best_params.json"))
    
    # Train final model with best params
    logger.info("\nTraining final model with best params...")
    train_data = lgb.Dataset(X_train, label=y_train, group=groups_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, group=groups_valid, reference=train_data)
    
    final_model = lgb.train(
        best_params,
        train_data,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
    )
    
    # Evaluate
    logger.info("\nEvaluation Results:")
    y_pred_valid = final_model.predict(X_valid)
    y_pred_test = final_model.predict(X_test)
    
    ndcg_valid = ndcg_at_k(y_valid, y_pred_valid, groups_valid, k=3)
    ndcg_test = ndcg_at_k(y_test, y_pred_test, groups_test, k=3)
    
    logger.info(f"Valid NDCG@3: {ndcg_valid:.4f}")
    logger.info(f"Test NDCG@3:  {ndcg_test:.4f}")
    
    # ROI calculation
    test_df = test_df.copy()
    test_df['pred'] = y_pred_test
    test_df['pred_rank'] = test_df.groupby('race_id')['pred'].rank(ascending=False)
    
    top1_bets = test_df[test_df['pred_rank'] == 1].copy()
    total_bet = len(top1_bets) * 100
    returns = top1_bets[top1_bets['is_win'] == 1]['odds'].sum() * 100
    roi = returns / total_bet * 100 if total_bet > 0 else 0
    
    logger.info(f"\nWin ROI (Bet Top 1): {roi:.2f}%")
    
    # Save model
    joblib.dump(final_model, os.path.join(OUTPUT_DIR, "model.pkl"))
    pd.Series(feature_cols).to_csv(os.path.join(OUTPUT_DIR, "features.csv"), index=False)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    importance.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    
    logger.info("\nTop 20 Features:")
    logger.info(importance.head(20).to_string())
    
    logger.info("=" * 60)
    logger.info("✅ Optuna HPO Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

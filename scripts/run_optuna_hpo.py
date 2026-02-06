"""
Optuna Hyperparameter Optimization for LightGBM
Usage: python scripts/run_optuna_hpo.py

Best parameters will be saved to config/experiments/exp_t2_optuna_best.yaml
"""
import os
import logging
import pickle
import yaml
import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = "data/processed/preprocessed_data_v11.parquet"
OUTPUT_DIR = "models/experiments/optuna_hpo"

def load_data(target='win'):
    """Load and prepare data for optimization"""
    logger.info(f"Loading data for target: {target}...")
    df = pd.read_parquet(DATA_PATH)
    
    # Load targets
    TARGET_PATH = "data/temp_t2/T2_targets.parquet"
    if os.path.exists(TARGET_PATH):
        logger.info(f"Loading targets from {TARGET_PATH}...")
        df_targets = pd.read_parquet(TARGET_PATH)
        
        # Ensure merge keys type match
        df['race_id'] = df['race_id'].astype(str)
        df_targets['race_id'] = df_targets['race_id'].astype(str)
        df['horse_number'] = pd.to_numeric(df['horse_number'], errors='coerce').fillna(0).astype(int)
        df_targets['horse_number'] = pd.to_numeric(df_targets['horse_number'], errors='coerce').fillna(0).astype(int)
        
        # Merge
        if 'date' in df_targets.columns:
            df = pd.merge(df, df_targets, on=['race_id', 'horse_number', 'date'], how='inner')
        else:
            df = pd.merge(df, df_targets, on=['race_id', 'horse_number'], how='inner')
        logger.info(f"Merged data shape: {df.shape}")
    else:
        logger.warning(f"Target file {TARGET_PATH} not found! Assuming 'rank' is in main data.")

    # Filter to 2019-2022 for training, 2023 for validation
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    df_train = df[(df['year'] >= 2019) & (df['year'] <= 2022)]
    df_val = df[df['year'] == 2023]
    
    # Sub-sampling for faster HPO on CPU (100k samples is enough for HPO)
    if len(df_train) > 100000:
        logger.info(f"Sub-sampling train data from {len(df_train)} to 100,000...")
        df_train = df_train.sample(n=100000, random_state=42)
    
    # Exclude meta, target and DIRECT LEAKAGE columns
    meta_cols = [
        'race_id', 'horse_number', 'date', 'rank', 'odds_final', 
        'is_win', 'is_top2', 'is_top3', 'year', 'rank_str', 'target'
    ]
    
    # Leakage found in diagnostic: Result-based columns
    leakage_cols = [
        'pass_1', 'pass_2', 'pass_3', 'pass_4', 'passing_rank',
        'last_3f', 'raw_time', 'time_diff', 'margin',
        'popularity', 'odds', 'relative_popularity_rank',
        'slow_start_recovery', 'track_bias_disadvantage',
        'outer_frame_disadv', 'wide_run', 'mean_time_diff_5', 'horse_wide_run_rate'
    ]
    
    exclude_all = meta_cols + leakage_cols
    feature_cols = [c for c in df.columns if c not in exclude_all]
    
    # Also exclude ID columns
    id_cols = ['horse_id', 'mare_id', 'sire_id', 'jockey_id', 'trainer_id']
    feature_cols = [c for c in feature_cols if c not in id_cols]
    
    X_train = df_train[feature_cols].copy()
    if target == 'top2':
        y_train = (df_train['rank'] <= 2).astype(int)
    elif target == 'top3':
        y_train = (df_train['rank'] <= 3).astype(int)
    else:
        y_train = (df_train['rank'] == 1).astype(int)
        
    X_val = df_val[feature_cols].copy()
    if target == 'top2':
        y_val = (df_val['rank'] <= 2).astype(int)
    elif target == 'top3':
        y_val = (df_val['rank'] <= 3).astype(int)
    else:
        y_val = (df_val['rank'] == 1).astype(int)
    
    # Handle non-numeric dtypes
    for col in X_train.columns:
        if X_train[col].dtype.name == 'category' or X_train[col].dtype == 'object':
            X_train[col] = X_train[col].astype('category').cat.codes
            X_val[col] = X_val[col].astype('category').cat.codes
        else:
            X_train[col] = X_train[col].fillna(-999)
            X_val[col] = X_val[col].fillna(-999)
    
    logger.info(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")
    logger.info(f"Features: {len(feature_cols)} columns")
    
    return X_train, y_train, X_val, y_val, feature_cols

def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function"""
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        
        # Hyperparameters to optimize
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    }
    
    # Create LightGBM datasets
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    # Train with early stopping
    callbacks = [
        lgb.early_stopping(stopping_rounds=30),
        # lgb.log_evaluation(period=10) 
    ]
    
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_val],
        callbacks=callbacks
    )
    
    # Return validation AUC
    y_pred = model.predict(X_val)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_val, y_pred)
    
    return auc

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="win", help="win, top2, top3")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info(f"Optuna Hyperparameter Optimization: Target={args.target}")
    logger.info("=" * 60)
    
    # Dynamic output directory
    output_dir = f"models/experiments/optuna_hpo_{args.target}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    X_train, y_train, X_val, y_val, feature_cols = load_data(args.target)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name=f'lgbm_hpo_{args.target}',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Run optimization
    n_trials = 10  # Quick test (set to 50 for full optimization)
    logger.info(f"Starting optimization with {n_trials} trials...")
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "=" * 60)
    print(" Optimization Results")
    print("=" * 60)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best AUC: {study.best_value:.5f}")
    
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Compare with current
    print("\n" + "-" * 60)
    print(" Comparison with Current Config")
    print("-" * 60)
    current = {
        'learning_rate': 0.05,
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
    }
    
    print(f"{'Parameter':<20} {'Current':>15} {'Optimized':>15}")
    print("-" * 50)
    for key in ['learning_rate', 'num_leaves', 'feature_fraction', 'bagging_fraction']:
        curr_val = current.get(key, 'N/A')
        opt_val = study.best_params.get(key, 'N/A')
        if isinstance(curr_val, float):
            print(f"{key:<20} {curr_val:>15.4f} {opt_val:>15.4f}")
        else:
            print(f"{key:<20} {curr_val:>15} {opt_val:>15}")
    
    # New params from optimization
    for key in ['max_depth', 'min_data_in_leaf', 'bagging_freq', 'lambda_l1', 'lambda_l2']:
        opt_val = study.best_params.get(key, 'N/A')
        if isinstance(opt_val, float) and opt_val < 0.01:
            print(f"{key:<20} {'N/A':>15} {opt_val:>15.2e}")
        elif isinstance(opt_val, float):
            print(f"{key:<20} {'N/A':>15} {opt_val:>15.4f}")
        else:
            print(f"{key:<20} {'N/A':>15} {opt_val:>15}")
    
    # Save best params
    best_config = {
        'experiment_name': 'exp_t2_optuna_best',
        'model_params': {
            'model_type': 'lightgbm',
            'objective': 'binary',
            'metric': 'auc',
            'n_estimators': 3000,
            'early_stopping_rounds': 100,
            'verbose': -1,
            'seed': 42,
            **study.best_params
        },
        'best_auc': study.best_value
    }
    
    config_path = os.path.join(output_dir, 'best_params.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)
    
    print(f"\nBest params saved to: {config_path}")
    
    # Save study
    study_path = os.path.join(output_dir, 'optuna_study.pkl')
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    print(f"Study saved to: {study_path}")

if __name__ == "__main__":
    main()

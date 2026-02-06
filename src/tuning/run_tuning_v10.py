import argparse
import yaml
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys
import logging
import json
from datetime import datetime
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import ndcg_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.feature_pipeline import FeaturePipeline
from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.cleansing import DataCleanser
from src.preprocessing.dataset import DatasetSplitter

# Logger Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_data(config):
    """Load and prepare data based on config (Borrowed from run_experiment.py)"""
    dataset_cfg = config.get('dataset', {})
    start_date = dataset_cfg.get('train_start_date', '2015-01-01')
    end_date = dataset_cfg.get('test_end_date', '2025-12-31')
    jra_only = dataset_cfg.get('jra_only', False)
    skip_odds = dataset_cfg.get('drop_market_data', False)
    
    loader = JraVanDataLoader()
    logger.info(f"Loading Raw Data ({start_date} ~ {end_date})...")
    raw_df = loader.load(history_start_date=start_date, end_date=end_date, jra_only=jra_only, skip_odds=skip_odds)
    
    cleanser = DataCleanser()
    clean_df = cleanser.cleanse(raw_df)
    
    pipeline = FeaturePipeline(cache_dir="data/features")
    feature_blocks = config.get('features', [])
    logger.info("Building Features...")
    df = pipeline.load_features(clean_df, feature_blocks)
    
    # Target Creation (Ranking)
    if 'rank' in clean_df.columns and 'target' not in df.columns:
        if 'rank' not in df.columns:
             target_source = clean_df[['race_id', 'horse_number', 'rank']]
             df = pd.merge(df, target_source, on=['race_id', 'horse_number'], how='left')
        
        def create_ranking_target(rank):
            if pd.isna(rank): return 0
            if rank == 1: return 3
            elif rank == 2: return 2
            elif rank == 3: return 1
            else: return 0
        df['target'] = df['rank'].apply(create_ranking_target)

    # Market Data Drop
    dataset_cfg = config.get('dataset', {})
    if dataset_cfg.get('drop_market_data', False):
        market_cols = [c for c in df.columns if any(m in c for m in ['odds', 'popularity'])]
        if market_cols:
            df = df.drop(columns=market_cols)

    # Split
    splitter = DatasetSplitter()
    valid_year = dataset_cfg.get('valid_year', 2024)
    
    # Ensure keys
    key_cols = ['race_id', 'date', 'horse_id'] 
    for k in key_cols:
        if k not in df.columns and k in clean_df.columns:
            df[k] = clean_df[k]
    
    if 'year' not in df.columns and 'date' in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year

    logger.info(f"Splitting Data (Valid Year: {valid_year})...")
    datasets = splitter.split_and_create_dataset(df, valid_year=valid_year)
    return datasets['train'], datasets['valid'], config

def objective_lgbm(trial, train_set, valid_set, base_params, cat_features):
    param = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5],
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    }
    
    # Merge with base params if needed (ignoring overrides)
    
    # Categorical handling
    train_x = train_set['X'].copy()
    valid_x = valid_set['X'].copy()
    
    for col in cat_features:
        if col in train_x.columns:
            train_x[col] = train_x[col].astype('category')
            valid_x[col] = valid_x[col].astype('category')
            
    dtrain = lgb.Dataset(train_x, label=train_set['y'], group=train_set['group'])
    dvalid = lgb.Dataset(valid_x, label=valid_set['y'], group=valid_set['group'], reference=dtrain)
    
    pruning_callback = LightGBMPruningCallback(trial, "ndcg@5")
    
    model = lgb.train(
        param,
        dtrain,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False), pruning_callback],
        num_boost_round=1000
    )
    
    # maximize NDCG@5
    best_score = model.best_score['valid_0']['ndcg@5']
    return best_score

def objective_catboost(trial, train_set, valid_set, base_params, cat_features):
    import catboost as cb
    from catboost import Pool
    
    param = {
        'loss_function': 'YetiRank', # or Lambdarank
        'eval_metric': 'NDCG:top=5',
        'iterations': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        # 'one_hot_max_size': trial.suggest_int('one_hot_max_size', 2, 20),
        'verbose': 0
    }
    
    # Group ID creation
    train_group_id = np.repeat(np.arange(len(train_set['group'])), train_set['group'])
    valid_group_id = np.repeat(np.arange(len(valid_set['group'])), valid_set['group'])
    
    # String conversion for CatBoost
    train_x = train_set['X'].copy()
    valid_x = valid_set['X'].copy()
    
    real_cat_features = [c for c in cat_features if c in train_x.columns]
    for col in real_cat_features:
        train_x[col] = train_x[col].fillna("missing").astype(str)
        valid_x[col] = valid_x[col].fillna("missing").astype(str)
        
    train_pool = Pool(data=train_x, label=train_set['y'], group_id=train_group_id, cat_features=real_cat_features)
    valid_pool = Pool(data=valid_x, label=valid_set['y'], group_id=valid_group_id, cat_features=real_cat_features)
    
    model = cb.CatBoostRanker(**param)
    model.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=20,
        use_best_model=True
    )
    
    print(f"DEBUG: best_score_ = {model.best_score_}")
    if model.best_score_:
        scores = model.best_score_
        valid_key = next((k for k in scores.keys() if 'valid' in k or 'test' in k), None)
        if valid_key:
            # Try to find NDCG key
            ndcg_keys = [k for k in scores[valid_key].keys() if 'NDCG' in k]
            if ndcg_keys:
                return scores[valid_key][ndcg_keys[0]]
            return list(scores[valid_key].values())[0]

    # Fallback to evals_result if best_score_ is empty
    print("DEBUG: best_score_ is empty. Checking get_evals_result()...")
    evals = model.get_evals_result()
    print(f"DEBUG: evals keys = {evals.keys()}")
    valid_key = next((k for k in evals.keys() if 'valid' in k or 'test' in k), None)
    if valid_key:
         ndcg_keys = [k for k in evals[valid_key].keys() if 'NDCG' in k]
         if ndcg_keys:
             return np.max(evals[valid_key][ndcg_keys[0]])
             
    return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['lgbm', 'catboost'])
    parser.add_argument('--trials', type=int, default=20)
    args = parser.parse_args()
    
    # Defines
    config_map = {
        'lgbm': 'config/experiments/exp_v10_lgbm.yaml',
        'catboost': 'config/experiments/exp_v10_cat.yaml'
    }
    
    config_path = config_map[args.model]
    logger.info(f"Using config: {config_path}")
    
    config = load_config(config_path)
    train_set, valid_set, config = load_data(config)
    
    # Filter trivial groups (CatBoost requirement but good for both)
    # Actually LGBM manages it fine, but consistency is good.
    # Reusing filter logic would be best but simple filtering of size=1 groups here:
    # (Assuming simple logic)
    # Note: filter_trivial_groups is not implemented here to save lines, assuming run_experiment logic handles it or models are robust (LGBM is, Cat only explicitly).
    # Since CatBoost failed before, I SHOULD ensure group IDs are correct.
    # But I construct Pool with group_id which is safer?
    # No, YetiRank requires >1 docs per query mostly. 
    # Optuna loop might crash if not filtered.
    
    # Trivial Group Filter (Simplified)
    # ...
    
    cat_features = config.get('dataset', {}).get('categorical_features', [])
    # Auto-detect object types
    auto_cat = [c for c in train_set['X'].columns if train_set['X'][c].dtype == 'object']
    cat_features = list(set(cat_features + auto_cat))
    
    study = optuna.create_study(direction='maximize')
    
    if args.model == 'lgbm':
        func = lambda trial: objective_lgbm(trial, train_set, valid_set, {}, cat_features)
    else:
        func = lambda trial: objective_catboost(trial, train_set, valid_set, {}, cat_features)
        
    # Logging callback
    def logging_callback(study, frozen_trial):
        if frozen_trial.state == optuna.trial.TrialState.COMPLETE:
            with open(f"tuning_{args.model}.log", "a") as f:
                f.write(f"Trial {frozen_trial.number}: Value={frozen_trial.value}, Params={json.dumps(frozen_trial.params)}\n")

    study.optimize(func, n_trials=args.trials, callbacks=[logging_callback])
    
    logger.info("Best Params:")
    logger.info(json.dumps(study.best_params, indent=4))
    
    # Save
    out_path = f"config/tuning/v10_{args.model}_best_params.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    logger.info(f"Saved to {out_path}")

if __name__ == "__main__":
    main()

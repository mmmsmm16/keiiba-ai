import sys
import os
import pickle
import argparse
import logging
import json
import optuna
import lightgbm as lgb
from catboost import CatBoostRanker, Pool
from pytorch_tabnet.tab_model import TabNetRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_datasets():
    dataset_path = os.path.join(os.path.dirname(__file__), '../../data/processed/lgbm_datasets.pkl')
    if not os.path.exists(dataset_path):
        logger.error(f"データセットが見つかりません: {dataset_path}")
        sys.exit(1)

    logger.info("データセットをロード中...")
    with open(dataset_path, 'rb') as f:
        datasets = pickle.load(f)
    
    return datasets['train'], datasets['valid']

def expand_group_ids(group_counts):
    """
    LightGBM用のgroup (各クエリのデータ数) を
    CatBoost用のgroup_id (各データのクエリID) に変換する
    """
    return np.repeat(np.arange(len(group_counts)), group_counts)

def prepare_tabnet_data(df):
    """
    TabNet用にDataFrameを数値numpy配列に変換する
    カテゴリ変数はコードに変換する
    """
    df_num = df.copy()
    for col in df_num.columns:
        if df_num[col].dtype.name == 'category':
            df_num[col] = df_num[col].cat.codes
        elif df_num[col].dtype == object:
            df_num[col] = df_num[col].astype('category').cat.codes
    return df_num.fillna(0).values

def objective_lgbm(trial, train_set, valid_set):
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    }

    lgb_train = lgb.Dataset(train_set['X'], label=train_set['y'], group=train_set['group'])
    lgb_valid = lgb.Dataset(valid_set['X'], label=valid_set['y'], group=valid_set['group'], reference=lgb_train)

    callbacks = [
        lgb.early_stopping(stopping_rounds=20, verbose=False),
        optuna.integration.LightGBMPruningCallback(trial, "ndcg@1")
    ]

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_valid],
        callbacks=callbacks
    )

    # Use NDCG@1 as optimization target (maximize)
    # LightGBM validation results are accessible via model.best_score['valid_0']['ndcg@1']
    return model.best_score['valid_0']['ndcg@1']

def objective_catboost(trial, train_set, valid_set):
    params = {
        'loss_function': 'YetiRank',
        'eval_metric': 'NDCG', # Explicitly optimize NDCG
        'iterations': 1000,
        'random_seed': 42,
        'verbose': False,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
    }

    # group (データ数) を group_id (各行のID) に変換
    train_group_id = expand_group_ids(train_set['group'])
    valid_group_id = expand_group_ids(valid_set['group'])

    train_pool = Pool(data=train_set['X'], label=train_set['y'], group_id=train_group_id)
    valid_pool = Pool(data=valid_set['X'], label=valid_set['y'], group_id=valid_group_id)

    model = CatBoostRanker(**params)
    model.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=20,
        use_best_model=True
    )
    
    # CatBoost optimization metric retrieval
    scores = model.get_best_score()
    best_score = 0.0
    
    # Method 1: get_best_score() (Standard dictionary)
    valid_key = next((k for k in scores.keys() if 'valid' in k or 'test' in k), None)
    if valid_key:
        ndcg_key = next((k for k in scores[valid_key].keys() if 'NDCG' in k), None)
        if ndcg_key:
            best_score = scores[valid_key][ndcg_key]
        elif 'PFound' in scores[valid_key]:
            best_score = scores[valid_key]['PFound']
        elif scores[valid_key]:
            best_score = list(scores[valid_key].values())[0]

    # Method 2: get_evals_result() (History fallback)
    if best_score == 0.0:
        evals = model.get_evals_result()
        # Look for validation set name
        for set_name in evals.keys():
             if 'valid' in set_name or 'test' in set_name:
                 # Look for NDCG metric
                 for metric in evals[set_name].keys():
                     if 'NDCG' in metric:
                         # Use max value from history (NDCG is maximized)
                         best_score = max(evals[set_name][metric])
                         break
                 if best_score > 0.0:
                     break
    
    return best_score

def objective_tabnet(trial, train_set, valid_set):
    params = {
        'optimizer_params': dict(lr=trial.suggest_float('lr', 1e-3, 1e-1, log=True)),
        'n_d': trial.suggest_int('n_d', 8, 64),
        'n_a': trial.suggest_int('n_a', 8, 64),
        'n_steps': trial.suggest_int('n_steps', 3, 10),
        'gamma': trial.suggest_float('gamma', 1.0, 2.0),
        'n_independent': trial.suggest_int('n_independent', 1, 5),
        'n_shared': trial.suggest_int('n_shared', 1, 5),
    }
    
    # Batch size
    batch_size = trial.suggest_categorical('batch_size', [1024, 2048, 4096])
    
    X_train = prepare_tabnet_data(train_set['X'])
    y_train = train_set['y'].values.reshape(-1, 1)
    X_valid = prepare_tabnet_data(valid_set['X'])
    y_valid = valid_set['y'].values.reshape(-1, 1)

    model = TabNetRegressor(
        **params,
        verbose=0,
        device_name='cpu' # Force CPU to avoid CUDA conflicts in docker
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_name=['valid'],
        eval_metric=['rmse'],
        max_epochs=50, # Limit epochs for speed
        patience=10,
        batch_size=batch_size,
        virtual_batch_size=128
    )
    
    # Minimize RMSE
    return model.best_cost

def main():
    parser = argparse.ArgumentParser(description='ハイパーパラメータ最適化スクリプト')
    parser.add_argument('--model', type=str, required=True, choices=['lgbm', 'catboost', 'tabnet'], help='Optimization target model')
    parser.add_argument('--trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--version', type=str, default='v1', help='Model version tag')
    args = parser.parse_args()

    # データロード (共通)
    logger.info("データをロード中...")
    train_set, valid_set = load_datasets()
    datasets = {'train': train_set, 'valid': valid_set}

    # 目的関数の作成
    if args.model == 'lgbm':
        objective_func = lambda trial: objective_lgbm(trial, datasets['train'], datasets['valid'])
        study_name = f"lgbm_optimization_{args.version}"
    elif args.model == 'catboost':
        objective_func = lambda trial: objective_catboost(trial, datasets['train'], datasets['valid'])
        study_name = f"catboost_optimization_{args.version}"
    elif args.model == 'tabnet':
        objective_func = lambda trial: objective_tabnet(trial, datasets['train'], datasets['valid'])
        study_name = f"tabnet_optimization_{args.version}"

    # Optunaの実行
    storage_dir = os.path.join(os.path.dirname(__file__), '../../models')
    os.makedirs(storage_dir, exist_ok=True)
    storage_name = f"sqlite:///{storage_dir}/optuna_{args.model}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize' if args.model != 'tabnet' else 'minimize', # TabNetはRMSE最小化
        storage=storage_name,
        load_if_exists=True
    )
    
    logger.info(f"最適化を開始します (Model: {args.model}, Trials: {args.trials}, Version: {args.version})")
    study.optimize(objective_func, n_trials=args.trials)

    print(f"Best trial:")
    print(f"  Value: {study.best_value}")
    print(f"  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # パラメータの保存
    output_dir = os.path.join(os.path.dirname(__file__), '../../models/params')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{args.model}_{args.version}_best_params.json')
    
    with open(output_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    
    logger.info(f"最適化完了。ベストパラメータを保存しました: {output_path}")

if __name__ == "__main__":
    main()

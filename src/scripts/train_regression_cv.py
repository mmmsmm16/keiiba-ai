"""
時系列CV - 回帰モデル (v23)

v13スタイルのグレード付きターゲットで学習する。
- 1着 = 1.0
- 2着 = 0.5
- 3着 = 0.25
- 4着以下 = 0.0
"""
import os
import sys
import pickle
import json
import logging
import argparse
from datetime import datetime

# パス設定
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 時系列CV Fold定義
FOLDS = [
    {"train_end": 2020, "valid_year": 2021, "name": "fold1"},
    {"train_end": 2021, "valid_year": 2022, "name": "fold2"},
    {"train_end": 2022, "valid_year": 2023, "name": "fold3"},
    {"train_end": 2023, "valid_year": 2024, "name": "fold4"},
]

# LightGBMパラメータ (v13/v7から最適化済み)
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 76,
    'learning_rate': 0.1576,
    'min_data_in_leaf': 53,
    'feature_fraction': 0.802,
    'bagging_fraction': 0.562,
    'bagging_freq': 7,
    'lambda_l1': 1.57e-05,
    'lambda_l2': 0.05,
    'random_state': 42,
    'verbose': -1
}

def load_dataset(dataset_suffix: str = "_v10_leakfix"):
    """データセットをロード"""
    parquet_path = os.path.join(project_root, f'data/processed/preprocessed_data{dataset_suffix}.parquet')
    logger.info(f"前処理済みデータをロード中: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    pickle_path = os.path.join(project_root, f'data/processed/lgbm_datasets{dataset_suffix}.pkl')
    logger.info(f"特徴量リストをロード中: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        datasets = pickle.load(f)
    
    feature_cols = datasets['train']['X'].columns.tolist()
    logger.info(f"データロード完了: {len(df)} 件, {len(feature_cols)} 特徴量")
    
    return df, feature_cols

def create_graded_target(rank):
    """グレード付きターゲットを作成"""
    # 1着=1.0, 2着=0.5, 3着=0.25, 4着以下=0.0
    if rank == 1:
        return 1.0
    elif rank == 2:
        return 0.5
    elif rank == 3:
        return 0.25
    else:
        return 0.0

def create_fold_dataset(df: pd.DataFrame, feature_cols: list, train_end: int, valid_year: int):
    """Fold用にデータセットを分割（回帰モード）"""
    train_df = df[df['year'] <= train_end].copy()
    valid_df = df[df['year'] == valid_year].copy()
    
    # グレード付きターゲット作成
    train_df['graded_target'] = train_df['rank'].apply(create_graded_target)
    valid_df['graded_target'] = valid_df['rank'].apply(create_graded_target)
    
    # 欠損特徴量は0埋め
    for col in feature_cols:
        if col not in train_df.columns:
            train_df[col] = 0
        if col not in valid_df.columns:
            valid_df[col] = 0
    
    X_train = train_df[feature_cols]
    y_train = train_df['graded_target'].values
    
    X_valid = valid_df[feature_cols]
    y_valid = valid_df['graded_target'].values
    
    fold_train = {'X': X_train, 'y': y_train}
    fold_valid = {'X': X_valid, 'y': y_valid}
    
    logger.info(f"Fold: Train={len(X_train)} (2013-{train_end}), Valid={len(X_valid)} ({valid_year})")
    
    return fold_train, fold_valid, valid_df

def train_fold(fold_train: dict, fold_valid: dict, fold_name: str, output_dir: str, version: str, skip_existing: bool = True):
    """1つのFoldを学習（回帰モード）"""
    fold_dir = os.path.join(output_dir, fold_name)
    os.makedirs(fold_dir, exist_ok=True)
    
    lgbm_path = os.path.join(fold_dir, f'lgbm_{version}.pkl')
    catboost_path = os.path.join(fold_dir, f'catboost_{version}.pkl')
    meta_path = os.path.join(fold_dir, f'meta_{version}.pkl')
    
    # 既存モデルチェック
    if skip_existing and os.path.exists(meta_path):
        logger.info(f"=== {fold_name} 既存モデルをロード（学習スキップ） ===")
        with open(lgbm_path, 'rb') as f:
            lgbm_model = pickle.load(f)
        with open(catboost_path, 'rb') as f:
            catboost_model = pickle.load(f)
        with open(meta_path, 'rb') as f:
            meta_model = pickle.load(f)
        return {
            'lgbm': lgbm_model,
            'catboost': catboost_model,
            'meta': meta_model,
            'paths': {'lgbm': lgbm_path, 'catboost': catboost_path, 'meta': meta_path}
        }
    
    logger.info(f"=== {fold_name} 学習開始 (回帰モード) ===")
    
    X_train = fold_train['X']
    y_train = fold_train['y']
    X_valid = fold_valid['X']
    y_valid = fold_valid['y']
    
    # LightGBM (回帰)
    logger.info(f"[{fold_name}] LightGBM (回帰) 学習中...")
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
    
    lgbm_model = lgb.train(
        LGBM_PARAMS,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_valid],
        callbacks=callbacks
    )
    
    with open(lgbm_path, 'wb') as f:
        pickle.dump(lgbm_model, f)
    logger.info(f"LightGBM保存: {lgbm_path}")
    
    # CatBoost (回帰)
    logger.info(f"[{fold_name}] CatBoost (回帰) 学習中...")
    
    # カテゴリ変数を特定
    cat_features = [col for col in X_train.columns if X_train[col].dtype == 'category']
    
    catboost_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.0758,
        depth=10,
        l2_leaf_reg=0.000143,
        bagging_temperature=0.5,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50
    )
    
    catboost_model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        cat_features=cat_features if cat_features else None
    )
    
    with open(catboost_path, 'wb') as f:
        pickle.dump(catboost_model, f)
    logger.info(f"CatBoost保存: {catboost_path}")
    
    # Meta Model (スタッキング)
    logger.info(f"[{fold_name}] Meta Model 学習中...")
    lgbm_pred_valid = lgbm_model.predict(X_valid)
    catboost_pred_valid = catboost_model.predict(X_valid)
    
    meta_X = np.column_stack([lgbm_pred_valid, catboost_pred_valid])
    meta_model = LinearRegression()
    meta_model.fit(meta_X, y_valid)
    
    logger.info(f"Meta Weights: lgbm={meta_model.coef_[0]:.4f}, catboost={meta_model.coef_[1]:.4f}, bias={meta_model.intercept_:.4f}")
    
    with open(meta_path, 'wb') as f:
        pickle.dump(meta_model, f)
    
    logger.info(f"=== {fold_name} 学習完了 ===")
    
    return {
        'lgbm': lgbm_model,
        'catboost': catboost_model,
        'meta': meta_model,
        'paths': {'lgbm': lgbm_path, 'catboost': catboost_path, 'meta': meta_path}
    }

def evaluate_fold(models: dict, valid_df: pd.DataFrame, feature_cols: list, fold_name: str, output_dir: str):
    """Foldの評価"""
    logger.info(f"=== {fold_name} 評価中 ===")
    
    # 欠損特徴量は0埋め
    for col in feature_cols:
        if col not in valid_df.columns:
            valid_df[col] = 0
    
    X_valid = valid_df[feature_cols]
    
    # 予測
    lgbm_pred = models['lgbm'].predict(X_valid)
    catboost_pred = models['catboost'].predict(X_valid)
    meta_X = np.column_stack([lgbm_pred, catboost_pred])
    scores = models['meta'].predict(meta_X)
    
    # 評価用DataFrame
    eval_df = valid_df[['race_id', 'rank']].copy()
    eval_df['score'] = scores
    
    # Top1的中率
    def calc_top1_accuracy(df):
        correct = 0
        total = 0
        for race_id, group in df.groupby('race_id'):
            top1_idx = group['score'].idxmax()
            if group.loc[top1_idx, 'rank'] == 1:
                correct += 1
            total += 1
        return correct / total if total > 0 else 0
    
    # Top3的中率（複勝）
    def calc_top3_in_top3(df):
        correct = 0
        total = 0
        for race_id, group in df.groupby('race_id'):
            top3_pred = group.nlargest(3, 'score')
            actual_top3 = group[group['rank'] <= 3]
            overlap = len(set(top3_pred.index) & set(actual_top3.index))
            if overlap >= 1:  # 1頭でも当たれば
                correct += 1
            total += 1
        return correct / total if total > 0 else 0
    
    top1_acc = calc_top1_accuracy(eval_df)
    top3_acc = calc_top3_in_top3(eval_df)
    
    results = {
        'fold': fold_name,
        'valid_size': len(eval_df),
        'top1_accuracy': top1_acc,
        'top3_in_top3': top3_acc,
    }
    
    logger.info(f"[{fold_name}] Top1的中率: {top1_acc:.2%}, Top3含み率: {top3_acc:.2%}")
    
    # 結果保存
    result_path = os.path.join(output_dir, fold_name, 'metrics.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='時系列CV (回帰モード)')
    parser.add_argument('--dataset_suffix', type=str, default='_v10_leakfix')
    parser.add_argument('--version', type=str, default='v23')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, 'experiments', f'{args.version}_regression_cv')
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("時系列CV (回帰モード) 学習開始")
    logger.info(f"Version: {args.version}")
    logger.info(f"Target: グレード付き (1着=1.0, 2着=0.5, 3着=0.25)")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 60)
    
    df, feature_cols = load_dataset(args.dataset_suffix)
    
    all_results = []
    
    for fold in FOLDS:
        fold_name = fold['name']
        train_end = fold['train_end']
        valid_year = fold['valid_year']
        
        fold_train, fold_valid, valid_df_fold = create_fold_dataset(df, feature_cols, train_end, valid_year)
        models = train_fold(fold_train, fold_valid, fold_name, args.output_dir, args.version)
        results = evaluate_fold(models, valid_df_fold, feature_cols, fold_name, args.output_dir)
        all_results.append(results)
    
    # サマリー
    logger.info("=" * 60)
    logger.info("全Fold結果サマリー")
    logger.info("=" * 60)
    
    for r in all_results:
        logger.info(f"{r['fold']}: Top1={r['top1_accuracy']:.2%}, Top3含={r['top3_in_top3']:.2%}")
    
    avg_top1 = np.mean([r['top1_accuracy'] for r in all_results])
    avg_top3 = np.mean([r['top3_in_top3'] for r in all_results])
    logger.info(f"平均: Top1={avg_top1:.2%}, Top3含={avg_top3:.2%}")
    
    # サマリー保存
    summary_path = os.path.join(args.output_dir, 'cv_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'version': args.version,
            'model_type': 'regression',
            'target': 'graded (1.0, 0.5, 0.25, 0.0)',
            'folds': all_results,
            'avg_top1_accuracy': avg_top1,
            'avg_top3_in_top3': avg_top3
        }, f, indent=2)
    
    logger.info(f"サマリー保存: {summary_path}")
    logger.info("時系列CV (回帰) 学習完了!")

if __name__ == "__main__":
    main()

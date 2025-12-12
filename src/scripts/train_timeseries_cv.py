"""
時系列Cross-Validation学習スクリプト

4つのFoldで順次学習し、各Foldで評価を行う。
- Fold 1: Train [2013-2020] → Valid [2021]
- Fold 2: Train [2013-2021] → Valid [2022]
- Fold 3: Train [2013-2022] → Valid [2023]
- Fold 4: Train [2013-2023] → Valid [2024]
- Final Test: [2025]
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

from model.lgbm import KeibaLGBM
from model.catboost_model import KeibaCatBoost
from model.ensemble import EnsembleModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 時系列CV Fold定義
FOLDS = [
    {"train_end": 2020, "valid_year": 2021, "name": "fold1"},
    {"train_end": 2021, "valid_year": 2022, "name": "fold2"},
    {"train_end": 2022, "valid_year": 2023, "name": "fold3"},
    {"train_end": 2023, "valid_year": 2024, "name": "fold4"},
]

def load_dataset(dataset_suffix: str = "_v10_leakfix"):
    """データセットをロード (preprocessed_dataとfeature_colsを取得)"""
    # Preprocessed data
    parquet_path = os.path.join(project_root, f'data/processed/preprocessed_data{dataset_suffix}.parquet')
    logger.info(f"前処理済みデータをロード中: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Feature columns from pickle
    pickle_path = os.path.join(project_root, f'data/processed/lgbm_datasets{dataset_suffix}.pkl')
    logger.info(f"特徴量リストをロード中: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        datasets = pickle.load(f)
    
    feature_cols = datasets['train']['X'].columns.tolist()
    
    logger.info(f"データロード完了: {len(df)} 件, {len(feature_cols)} 特徴量")
    
    return df, feature_cols

def create_fold_dataset(df: pd.DataFrame, feature_cols: list, train_end: int, valid_year: int):
    """Fold用にデータセットを分割"""
    # 年でフィルタ
    train_df = df[df['year'] <= train_end].copy()
    valid_df = df[df['year'] == valid_year].copy()
    
    # race_idでソート (LambdaRankで必要)
    train_df = train_df.sort_values(['race_id']).reset_index(drop=True)
    valid_df = valid_df.sort_values(['race_id']).reset_index(drop=True)
    
    # 特徴量とターゲットの分離
    # 欠損特徴量は0埋め
    for col in feature_cols:
        if col not in train_df.columns:
            train_df[col] = 0
        if col not in valid_df.columns:
            valid_df[col] = 0
    
    X_train = train_df[feature_cols]
    y_train = train_df['rank'].values
    
    X_valid = valid_df[feature_cols]
    y_valid = valid_df['rank'].values
    
    # LambdaRank用: groupはレースごとの馬数（クエリサイズ）の配列
    # race_idでgroupbyしてカウント
    train_group_sizes = train_df.groupby('race_id').size().values
    valid_group_sizes = valid_df.groupby('race_id').size().values
    
    fold_train = {'X': X_train, 'y': y_train, 'group': train_group_sizes}
    fold_valid = {'X': X_valid, 'y': y_valid, 'group': valid_group_sizes}
    
    logger.info(f"Fold: Train={len(X_train)} ({len(train_group_sizes)} races, 2013-{train_end}), Valid={len(X_valid)} ({len(valid_group_sizes)} races, {valid_year})")
    
    return fold_train, fold_valid

def train_fold(fold_train: dict, fold_valid: dict, fold_name: str, output_dir: str, version: str):
    """1つのFoldを学習"""
    logger.info(f"=== {fold_name} 学習開始 ===")
    
    fold_dir = os.path.join(output_dir, fold_name)
    os.makedirs(fold_dir, exist_ok=True)
    
    # LightGBM
    logger.info(f"[{fold_name}] LightGBM学習中...")
    lgbm = KeibaLGBM()
    lgbm.train(fold_train, fold_valid)
    lgbm_path = os.path.join(fold_dir, f'lgbm_{version}.pkl')
    lgbm.save_model(lgbm_path)
    
    # CatBoost
    logger.info(f"[{fold_name}] CatBoost学習中...")
    catboost = KeibaCatBoost()
    catboost.train(fold_train, fold_valid)
    catboost_path = os.path.join(fold_dir, f'catboost_{version}.pkl')
    catboost.save_model(catboost_path)
    
    # Ensemble Meta-Model
    logger.info(f"[{fold_name}] Ensemble学習中...")
    ensemble = EnsembleModel()
    ensemble.lgbm = lgbm
    ensemble.catboost = catboost
    ensemble.has_lgbm = True
    ensemble.has_catboost = True
    ensemble.has_tabnet = False
    ensemble.train_meta_model(fold_valid)
    ensemble_path = os.path.join(fold_dir, f'ensemble_{version}.pkl')
    ensemble.save_model(ensemble_path)
    
    logger.info(f"=== {fold_name} 学習完了 ===")
    
    return {
        'lgbm': lgbm,
        'catboost': catboost,
        'ensemble': ensemble,
        'paths': {
            'lgbm': lgbm_path,
            'catboost': catboost_path,
            'ensemble': ensemble_path
        }
    }

def evaluate_fold(models: dict, fold_valid: dict, fold_name: str, output_dir: str):
    """Foldの評価を実行"""
    from scipy.special import softmax
    
    logger.info(f"=== {fold_name} 評価中 ===")
    
    X_valid = fold_valid['X']
    y_valid = fold_valid['y']
    group_valid = fold_valid['group']
    
    # 予測
    scores = models['ensemble'].predict(X_valid)
    
    # DataFrameにまとめる
    eval_df = pd.DataFrame({
        'race_id': group_valid,
        'rank': y_valid,
        'score': scores
    })
    
    # レースごとのTop1的中率
    def calc_top1_accuracy(df):
        correct = 0
        total = 0
        for race_id, group in df.groupby('race_id'):
            top1_idx = group['score'].idxmax()
            if group.loc[top1_idx, 'rank'] == 1:
                correct += 1
            total += 1
        return correct / total if total > 0 else 0
    
    top1_accuracy = calc_top1_accuracy(eval_df)
    
    results = {
        'fold': fold_name,
        'valid_size': len(eval_df),
        'top1_accuracy': top1_accuracy,
    }
    
    logger.info(f"[{fold_name}] Top1的中率: {top1_accuracy:.2%}")
    
    # 結果を保存
    result_path = os.path.join(output_dir, fold_name, 'metrics.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='時系列CV学習スクリプト')
    parser.add_argument('--dataset_suffix', type=str, default='_v10_leakfix', help='データセットのサフィックス')
    parser.add_argument('--version', type=str, default='v22', help='モデルバージョン')
    parser.add_argument('--output_dir', type=str, default=None, help='出力ディレクトリ')
    args = parser.parse_args()
    
    # 出力ディレクトリ
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, 'experiments', f'{args.version}_timeseries_cv')
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("時系列Cross-Validation 学習開始")
    logger.info(f"Version: {args.version}")
    logger.info(f"Dataset: lgbm_datasets{args.dataset_suffix}.pkl")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 60)
    
    # データセットロード
    df, feature_cols = load_dataset(args.dataset_suffix)
    
    all_results = []
    
    for fold in FOLDS:
        fold_name = fold['name']
        train_end = fold['train_end']
        valid_year = fold['valid_year']
        
        # Fold用データセット作成
        fold_train, fold_valid = create_fold_dataset(df, feature_cols, train_end, valid_year)
        
        # 学習
        models = train_fold(fold_train, fold_valid, fold_name, args.output_dir, args.version)
        
        # 評価
        results = evaluate_fold(models, fold_valid, fold_name, args.output_dir)
        all_results.append(results)
    
    # 全Fold結果サマリー
    logger.info("=" * 60)
    logger.info("全Fold結果サマリー")
    logger.info("=" * 60)
    
    for r in all_results:
        logger.info(f"{r['fold']}: Top1的中率={r['top1_accuracy']:.2%}")
    
    avg_accuracy = np.mean([r['top1_accuracy'] for r in all_results])
    logger.info(f"平均Top1的中率: {avg_accuracy:.2%}")
    
    # サマリー保存
    summary_path = os.path.join(args.output_dir, 'cv_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'version': args.version,
            'dataset_suffix': args.dataset_suffix,
            'folds': all_results,
            'avg_top1_accuracy': avg_accuracy
        }, f, indent=2)
    
    logger.info(f"サマリー保存: {summary_path}")
    logger.info("時系列CV 学習完了!")

if __name__ == "__main__":
    main()

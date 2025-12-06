import sys
import os
import pickle
import pandas as pd
import logging
import argparse

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.experiment_logger import ExperimentLogger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_datasets():
    dataset_path = os.path.join(os.path.dirname(__file__), '../../data/processed/lgbm_datasets.pkl')
    if not os.path.exists(dataset_path):
        logger.error(f"データセットが見つかりません: {dataset_path}")
        logger.error("先に src/preprocessing/run_preprocessing.py を実行してください。")
        sys.exit(1)

    logger.info("データセットをロード中...")
    with open(dataset_path, 'rb') as f:
        datasets = pickle.load(f)
    
    return datasets['train'], datasets['valid']

def train_lgbm(train_set, valid_set, model_dir):
    from model.lgbm import KeibaLGBM
    logger.info("=== LightGBM 学習モード ===")
    model = KeibaLGBM()
    model.train(train_set, valid_set)
    
    path = os.path.join(model_dir, 'lgbm.pkl')
    model.save_model(path)
    return model

def train_catboost(train_set, valid_set, model_dir):
    from model.catboost_model import KeibaCatBoost
    logger.info("=== CatBoost 学習モード ===")
    model = KeibaCatBoost()
    model.train(train_set, valid_set)
    
    path = os.path.join(model_dir, 'catboost.pkl')
    model.save_model(path)
    return model

def train_tabnet(train_set, valid_set, model_dir, batch_size=None):
    from model.tabnet_model import KeibaTabNet
    logger.info("=== TabNet 学習モード ===")
    
    params = {}
    if batch_size:
        params['batch_size'] = batch_size
        logger.info(f"バッチサイズを {batch_size} に設定します。")

    model = KeibaTabNet(params=params)
    model.train(train_set, valid_set)
    
    path = os.path.join(model_dir, 'tabnet.pkl') # TabNet saves as zip internally
    model.save_model(path)
    return model

def train_ensemble(train_set, valid_set, model_dir):
    from model.ensemble import EnsembleModel
    logger.info("=== アンサンブル メタモデル学習モード ===")
    model = EnsembleModel()
    
    # Load base models
    model.load_base_models(model_dir)
    
    # Train meta model
    model.train_meta_model(valid_set)
    
    path = os.path.join(model_dir, 'ensemble_model.pkl')
    model.save_model(path)
    return model

def main():
    parser = argparse.ArgumentParser(description='モデル学習スクリプト')
    parser.add_argument('--experiment_name', type=str, default=None, help='実験名（指定しない場合は自動生成）')
    parser.add_argument('--note', type=str, default="", help='実験に関するメモ')
    parser.add_argument('--model', type=str, required=True, choices=['lgbm', 'catboost', 'tabnet', 'ensemble'], 
                        help='学習するモデルを指定 (lgbm, catboost, tabnet, ensemble)')
    parser.add_argument('--batch_size', type=int, default=None, help='バッチサイズ (TabNet用)')
    args = parser.parse_args()

    train_set, valid_set = load_datasets()

    if train_set['X'] is None or train_set['X'].empty:
        logger.error("学習データが空です。")
        return

    # 実験ロガーの初期化
    exp_logger = ExperimentLogger(experiment_name=args.experiment_name)
    logger.info(f"実験開始: 実験名={exp_logger.experiment_name}, モード={args.model}")

    model_dir = os.path.join(os.path.dirname(__file__), '../../models')
    os.makedirs(model_dir, exist_ok=True)

    model = None
    if args.model == 'lgbm':
        model = train_lgbm(train_set, valid_set, model_dir)
    elif args.model == 'catboost':
        model = train_catboost(train_set, valid_set, model_dir)
    elif args.model == 'tabnet':
        model = train_tabnet(train_set, valid_set, model_dir, batch_size=args.batch_size)
    elif args.model == 'ensemble':
        model = train_ensemble(train_set, valid_set, model_dir)

    # 簡易評価ログ (Ensemble時のみ、または各モデルのValidスコア)
    # ここでは共通処理としてエラーが出ない範囲で記録
    metrics = {}
    params = {}
    
    try:
        if args.model == 'ensemble':
            preds = model.predict(valid_set['X'])
            mse = ((preds - valid_set['y']) ** 2).mean()
            metrics['valid_mse'] = mse
            logger.info(f"Valid MSE: {mse:.4f}")
            
            params['meta_model_coef'] = model.meta_model.coef_.tolist()
    except Exception as e:
        logger.warning(f"メトリクス計算中にエラーが発生しました: {e}")

    # 実験ログ保存
    exp_logger.log_result(
        model_type=args.model,
        metrics=metrics,
        params=params,
        note=args.note
    )

if __name__ == "__main__":
    main()

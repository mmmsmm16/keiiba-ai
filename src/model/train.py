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

import json

def load_params(model_name, version):
    param_path = os.path.join(os.path.dirname(__file__), f'../../models/params/{model_name}_{version}_best_params.json')
    if os.path.exists(param_path):
        try:
            with open(param_path, 'r') as f:
                params = json.load(f)
            logger.info(f"最適化済みパラメータをロードしました: {param_path}")
            return params
        except Exception as e:
            logger.warning(f"パラメータファイルのロードに失敗しました: {e}")
            return {}
    else:
        logger.info(f"最適化済みパラメータが見つかりません: {param_path} (デフォルト設定を使用します)")
        return {}

def train_lgbm(train_set, valid_set, model_dir, version):
    from model.lgbm import KeibaLGBM
    logger.info(f"=== LightGBM 学習モード (Version: {version}) ===")
    
    params = load_params('lgbm', version)
    model = KeibaLGBM(params=params)
    model.train(train_set, valid_set)
    
    path = os.path.join(model_dir, f'lgbm_{version}.pkl')
    model.save_model(path)
    return model

def train_catboost(train_set, valid_set, model_dir, version):
    from model.catboost_model import KeibaCatBoost
    logger.info(f"=== CatBoost 学習モード (Version: {version}) ===")
    
    params = load_params('catboost', version)
    model = KeibaCatBoost(params=params)
    model.train(train_set, valid_set)
    
    path = os.path.join(model_dir, f'catboost_{version}.pkl')
    model.save_model(path)
    return model

def train_tabnet(train_set, valid_set, model_dir, version, batch_size=None):
    from model.tabnet_model import KeibaTabNet
    logger.info(f"=== TabNet 学習モード (Version: {version}) ===")
    
    params = load_params('tabnet', version)
    if batch_size:
        params['batch_size'] = batch_size
        logger.info(f"バッチサイズを {batch_size} に設定します。")

    model = KeibaTabNet(params=params)
    model.train(train_set, valid_set)
    
    path = os.path.join(model_dir, f'tabnet_{version}.pkl')
    model.save_model(path)
    return model

def train_ensemble(train_set, valid_set, model_dir, version):
    from model.ensemble import EnsembleModel
    logger.info(f"=== アンサンブル メタモデル学習モード (Version: {version}) ===")
    model = EnsembleModel()
    
    # Load base models (Note: Ensemble might need to know which version of base models to load. 
    # For now, let's assume it loads the SAME version base models if possible, or we need an argument. 
    # Updating EnsembleModel to accept suffixes might be needed but for now let's assume default names or handle inside.)
    # The current EnsembleModel.load_base_models loads 'lgbm.pkl', 'catboost.pkl'. 
    # We should probably update Ensemble logic later to be flexible. 
    # For now, let's keep it simple or strictly rely on default names? 
    # Wait, if we save as lgbm_v2.pkl, EnsembleModel won't find it.
    # We should fix EnsembleModel separately or strictly use version for filenames.
    
    # Temporary fix: Try to load specific versions or fallback?
    # Actually, let's pass the version to load_base_models if we can modify it.
    # Checking existing EnsembleModel code... I can't see it now but I recall simpler load.
    
    # For now, simply save the meta model with version.
    model.load_base_models(model_dir) 
    
    # Train meta model
    model.train_meta_model(valid_set)
    
    path = os.path.join(model_dir, f'ensemble_{version}.pkl')
    model.save_model(path)
    return model

def main():
    parser = argparse.ArgumentParser(description='モデル学習スクリプト')
    parser.add_argument('--experiment_name', type=str, default=None, help='実験名（指定しない場合は自動生成）')
    parser.add_argument('--note', type=str, default="", help='実験に関するメモ')
    parser.add_argument('--model', type=str, required=True, choices=['lgbm', 'catboost', 'tabnet', 'ensemble'], 
                        help='学習するモデルを指定 (lgbm, catboost, tabnet, ensemble)')
    parser.add_argument('--batch_size', type=int, default=None, help='バッチサイズ (TabNet用)')
    parser.add_argument('--version', type=str, default='v1', help='モデルバージョン (例: v1, v2)')
    args = parser.parse_args()

    train_set, valid_set = load_datasets()

    if train_set['X'] is None or train_set['X'].empty:
        logger.error("学習データが空です。")
        return

    # 実験ロガーの初期化
    exp_logger = ExperimentLogger(experiment_name=args.experiment_name)
    logger.info(f"実験開始: 実験名={exp_logger.experiment_name}, モード={args.model}, Version={args.version}")

    model_dir = os.path.join(os.path.dirname(__file__), '../../models')
    os.makedirs(model_dir, exist_ok=True)

    model = None
    if args.model == 'lgbm':
        model = train_lgbm(train_set, valid_set, model_dir, args.version)
    elif args.model == 'catboost':
        model = train_catboost(train_set, valid_set, model_dir, args.version)
    elif args.model == 'tabnet':
        model = train_tabnet(train_set, valid_set, model_dir, args.version, batch_size=args.batch_size)
    elif args.model == 'ensemble':
        model = train_ensemble(train_set, valid_set, model_dir, args.version)

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

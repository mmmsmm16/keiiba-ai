import sys
import os
import pickle
import pandas as pd
import logging
import argparse

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.ensemble import EnsembleModel
from utils.experiment_logger import ExperimentLogger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='モデル学習スクリプト')
    parser.add_argument('--experiment_name', type=str, default=None, help='実験名（指定しない場合は自動生成）')
    parser.add_argument('--note', type=str, default="", help='実験に関するメモ')
    args = parser.parse_args()

    # データセットのロード
    dataset_path = os.path.join(os.path.dirname(__file__), '../../data/processed/lgbm_datasets.pkl')
    if not os.path.exists(dataset_path):
        logger.error(f"データセットが見つかりません: {dataset_path}")
        logger.error("先に src/preprocessing/run_preprocessing.py を実行してください。")
        return

    logger.info("データセットをロード中...")
    with open(dataset_path, 'rb') as f:
        datasets = pickle.load(f)

    train_set = datasets['train']
    valid_set = datasets['valid']

    if train_set['X'] is None or train_set['X'].empty:
        logger.error("学習データが空です。")
        return

    # 実験ロガーの初期化
    exp_logger = ExperimentLogger(experiment_name=args.experiment_name)

    # アンサンブルモデル学習 (LightGBM + CatBoost + TabNet)
    logger.info(f"学習開始: 実験名={exp_logger.experiment_name}")
    model = EnsembleModel()

    # 学習実行
    model.train(train_set, valid_set)

    # モデル保存
    model_dir = os.path.join(os.path.dirname(__file__), '../../models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'ensemble_model.pkl')
    model.save_model(model_path)

    # 簡易評価（バリデーションスコアの記録）
    # EnsembleModelのtrain()内でMetaModelの学習が行われるが、
    # ここではMetaModelの係数などをパラメータとして記録する。

    params = {
        'lgbm_params': model.lgbm.params,
        'catboost_params': model.catboost.params,
        'tabnet_params': model.tabnet.params,
        'meta_model_coef': model.meta_model.coef_.tolist(),
        'meta_model_intercept': model.meta_model.intercept_
    }

    # 指標の計算 (Valid setに対するMSEなど、簡易的なもの)
    preds = model.predict(valid_set['X'])
    mse = ((preds - valid_set['y']) ** 2).mean()
    metrics = {'valid_mse': mse}

    logger.info(f"Valid MSE: {mse:.4f}")

    # 実験ログ保存
    exp_logger.log_result(
        model_type="Ensemble_LGBM_Cat_TabNet",
        metrics=metrics,
        params=params,
        note=args.note
    )

if __name__ == "__main__":
    main()

import sys
import os
import pickle
import pandas as pd
import logging

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.lgbm import KeibaLGBM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
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

    # モデル学習
    model = KeibaLGBM()
    model.train(train_set, valid_set)

    # モデル保存
    model_path = os.path.join(os.path.dirname(__file__), '../../models/lgbm_model.pkl')
    model.save_model(model_path)

    # 特徴量重要度のプロット
    imp_path = os.path.join(os.path.dirname(__file__), '../../models/importance.png')
    model.plot_importance(imp_path)

if __name__ == "__main__":
    main()

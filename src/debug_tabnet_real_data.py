import sys
import os
import pickle
import pandas as pd
import logging
import torch
import numpy as np

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.tabnet_model import KeibaTabNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print(f"Torch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    # データセットのロード
    dataset_path = os.path.join(os.path.dirname(__file__), '../data/processed/lgbm_datasets.pkl')
    if not os.path.exists(dataset_path):
        logger.error(f"データセットが見つかりません: {dataset_path}")
        return

    logger.info("データセットをロード中...")
    with open(dataset_path, 'rb') as f:
        datasets = pickle.load(f)

    train_set = datasets['train']
    valid_set = datasets['valid']
    
    X_train = train_set['X']
    y_train = train_set['y']
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    # Check for NaNs or Infs
    if isinstance(X_train, pd.DataFrame):
        print(f"NaN count in X: {X_train.isna().sum().sum()}")
    
    logger.info("KeibaTabNetの初期化...")
    # Initialize with the same params as in the app
    model = KeibaTabNet()
    
    # Force reduced batch size just in case
    # model.fit_params['batch_size'] = 256 

    logger.info("学習開始 (Isolation Test)...")
    try:
        model.train(train_set, valid_set)
        logger.info("学習完了 (Success)")
    except Exception as e:
        logger.error(f"学習失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

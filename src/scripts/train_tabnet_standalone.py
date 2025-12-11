#!/usr/bin/env python
"""
TabNet単体学習スクリプト（GPU対応・進捗表示版）
デバッグ用にフラッシュ出力追加
"""
import os
import sys
import pickle
import logging
import torch
import numpy as np
import pandas as pd

# stdoutをアンバッファにする
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 設定
    experiment_dir = "experiments/v12_tabnet_revival"
    dataset_path = os.path.join(experiment_dir, "data/lgbm_datasets.pkl")
    output_path = os.path.join(experiment_dir, "models/tabnet.zip")
    
    # CUDA確認
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Device: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    
    # データ読み込み
    logger.info(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        datasets = pickle.load(f)
    
    train_set = datasets['train']
    valid_set = datasets['valid']
    
    logger.info(f"Train: {len(train_set['X'])} rows, Valid: {len(valid_set['X'])} rows")
    
    # TabNet学習
    from src.model.tabnet_model import KeibaTabNet
    
    params = {
        'enabled': True,
        'max_epochs': 50,
        'batch_size': 1024,
        'device_name': 'cuda',
        'patience': 10,
        'verbose': 1
    }
    
    logger.info("Creating TabNet model...")
    model = KeibaTabNet(params=params)
    
    logger.info("Starting training with full dataset...")
    print(">>> Training started", flush=True)
    model.train(train_set, valid_set)
    print(">>> Training finished", flush=True)
    
    # CUDA同期
    if torch.cuda.is_available():
        logger.info("Synchronizing CUDA...")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.info("CUDA synchronized")
    
    logger.info(f"Saving model to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save_model(output_path)
    logger.info("✅ TabNet training completed successfully!")

if __name__ == "__main__":
    main()

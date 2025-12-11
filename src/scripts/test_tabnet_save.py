#!/usr/bin/env python
"""
TabNet保存テスト用スクリプト（小規模データ）
モデル保存が正常に動作することを確認する
"""
import os
import sys
import pickle
import logging
import torch
import numpy as np
import pandas as pd

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 設定
    experiment_dir = "experiments/v12_tabnet_revival"
    dataset_path = os.path.join(experiment_dir, "data/lgbm_datasets.pkl")
    test_output_path = os.path.join(experiment_dir, "models/tabnet_test.zip")
    
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    
    # データ読み込み
    logger.info(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        datasets = pickle.load(f)
    
    # 10Kサンプルのみ使用
    SAMPLE_SIZE = 10000
    train_set = {
        'X': datasets['train']['X'].iloc[:SAMPLE_SIZE].copy(),
        'y': datasets['train']['y'][:SAMPLE_SIZE].copy()
    }
    valid_set = {
        'X': datasets['valid']['X'].iloc[:1000].copy(),
        'y': datasets['valid']['y'][:1000].copy()
    }
    
    logger.info(f"Test data: Train={len(train_set['X'])} rows, Valid={len(valid_set['X'])} rows")
    
    from src.model.tabnet_model import KeibaTabNet
    
    params = {
        'enabled': True,
        'max_epochs': 5,  # 短いテスト
        'batch_size': 512,
        'device_name': 'cuda',
        'patience': 3,
        'verbose': 1
    }
    
    logger.info(">>> Creating model...")
    model = KeibaTabNet(params=params)
    
    logger.info(">>> Starting training (5 epochs)...")
    model.train(train_set, valid_set)
    logger.info(">>> Training completed!")
    
    # CUDA同期
    if torch.cuda.is_available():
        logger.info(">>> Synchronizing CUDA...")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # モデル保存テスト
    logger.info(f">>> Saving model to: {test_output_path}")
    os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
    model.save_model(test_output_path)
    
    # 保存確認 (実際のファイルは base + .zip)
    actual_model_path = test_output_path.replace('.zip', '') + '.zip'
    if os.path.exists(actual_model_path):
        size = os.path.getsize(actual_model_path)
        logger.info(f"✅ Model saved successfully! Path: {actual_model_path}, Size: {size} bytes")
        
        # 読み込みテスト
        logger.info(">>> Testing model load...")
        model2 = KeibaTabNet(params={'enabled': True, 'device_name': 'cuda'})
        model2.load_model(test_output_path)
        logger.info("✅ Model loaded successfully!")
        
        # 推論テスト
        logger.info(">>> Testing prediction...")
        preds = model2.predict(valid_set['X'])
        logger.info(f"✅ Prediction test passed! Shape: {preds.shape}")
        
        # クリーンアップ
        os.remove(actual_model_path)
        scaler_path = test_output_path.replace('.zip', '') + '.scaler'
        if os.path.exists(scaler_path):
            os.remove(scaler_path)
        logger.info(">>> Test files cleaned up")
        
        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED! Ready for full training.")
        print("="*50)
        return True
    else:
        logger.error(f"❌ Model file was not created at: {actual_model_path}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

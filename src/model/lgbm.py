import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import logging

logger = logging.getLogger(__name__)

class KeibaLGBM:
    """
    LightGBM (LambdaRank) を使用した競馬予測モデルクラス。
    """
    def __init__(self, params=None):
        # デフォルトパラメータ (LambdaRank用)
        self.params = params or {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 3, 5],
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'random_state': 42,
            'verbose': -1
        }
        self.model = None

    def train(self, train_set: dict, valid_set: dict):
        """
        モデルを学習させます。

        Args:
            train_set (dict): {'X': DataFrame, 'y': Series, 'group': Array}
            valid_set (dict): {'X': DataFrame, 'y': Series, 'group': Array}
        """
        logger.info("LightGBMの学習を開始します...")

        # データセットの作成
        lgb_train = lgb.Dataset(train_set['X'], label=train_set['y'], group=train_set['group'])
        lgb_valid = lgb.Dataset(valid_set['X'], label=valid_set['y'], group=valid_set['group'], reference=lgb_train)

        # コールバック設定
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]

        self.model = lgb.train(
            self.params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_train, lgb_valid],
            callbacks=callbacks
        )
        logger.info("学習が完了しました。")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        推論を行います。
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません。")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def save_model(self, path: str):
        """モデルをpickle形式で保存します。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"モデルを保存しました: {path}")

    def load_model(self, path: str):
        """モデルをロードします。"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"モデルをロードしました: {path}")

    def plot_importance(self, output_path: str):
        """特徴量重要度をプロットして保存します。"""
        if not self.model:
            return
        plt.figure(figsize=(10, 6))
        lgb.plot_importance(self.model, max_num_features=20, figsize=(10, 6))
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"特徴量重要度を保存しました: {output_path}")

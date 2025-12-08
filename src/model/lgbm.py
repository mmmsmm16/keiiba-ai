import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import logging
import json

logger = logging.getLogger(__name__)

class KeibaLGBM:
    """
    LightGBM (LambdaRank) を使用した競馬予測モデルクラス。
    """
    def __init__(self, params=None):
        # デフォルトパラメータ (LambdaRank用)
        self.params = {
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
        
        # 引数指定があれば上書き
        if params:
            self.params.update(params)

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
        w_train = train_set.get('w')
        w_valid = valid_set.get('w')
        
        lgb_train = lgb.Dataset(train_set['X'], label=train_set['y'], group=train_set['group'], weight=w_train)
        lgb_valid = lgb.Dataset(valid_set['X'], label=valid_set['y'], group=valid_set['group'], reference=lgb_train, weight=w_valid)

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
        
        # 特徴量のフィルタリング (学習時に使用した特徴量のみに絞る)
        if hasattr(self.model, 'feature_name'):
            required_features = self.model.feature_name()
            # 必要なカラムが足りているかチェック
            missing = set(required_features) - set(X.columns)
            if missing:
                # 欠損がある場合はエラーにするか、NaNで埋めるかだが、通常はエラーが望ましい
                # しかしDashboard等で一部欠損許容するなら警告など。ここでは厳密にチェック。
                pass # LightGBM本体がエラーを出すのでそのままにする、あるいは独自エラー出す
            
            # 余分なカラムがある場合は削除して、順序を合わせる
            if len(X.columns) != len(required_features) or list(X.columns) != required_features:
                 # logger.debug("入力データの特徴量を学習時の形式に合わせます。")
                 X = X[required_features]

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

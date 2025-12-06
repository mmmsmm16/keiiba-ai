import catboost as cb
import pandas as pd
import numpy as np
import os
import pickle
import logging

logger = logging.getLogger(__name__)

class KeibaCatBoost:
    """
    CatBoost (YetiRank) を使用した競馬予測モデルクラス。
    """
    def __init__(self, params=None):
        self.params = params or {
            'loss_function': 'YetiRank',
            'eval_metric': 'NDCG',
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'random_seed': 42,
            'verbose': 100,
            'early_stopping_rounds': 50,
            'allow_writing_files': False, # 不要なログファイルの生成を防ぐ
            'task_type': 'CPU' # GPUリソース競合を避けるためCPUで実行
        }
        self.model = None

    def train(self, train_set: dict, valid_set: dict):
        """
        モデルを学習させます。

        Args:
            train_set (dict): {'X': DataFrame, 'y': Series, 'group': Array(counts)}
            valid_set (dict): {'X': DataFrame, 'y': Series, 'group': Array(counts)}
        """
        logger.info("CatBoostの学習を開始します...")

        # CatBoost Poolの作成 (group_idが必要)
        train_pool = cb.Pool(
            data=train_set['X'],
            label=train_set['y'],
            group_id=self._create_group_id(train_set['group'])
        )
        valid_pool = cb.Pool(
            data=valid_set['X'],
            label=valid_set['y'],
            group_id=self._create_group_id(valid_set['group'])
        )

        self.model = cb.CatBoostRanker(**self.params)
        self.model.fit(train_pool, eval_set=valid_pool)
        logger.info("学習が完了しました。")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        推論を行います。
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません。")
        return self.model.predict(X)

    def _create_group_id(self, group_counts):
        """
        グループカウント配列 ([12, 14]) をグループID配列 ([0...0, 1...1]) に変換します。
        CatBoost Pool は group_id を要求するため。
        """
        group_ids = []
        for i, count in enumerate(group_counts):
            group_ids.extend([i] * int(count))
        return group_ids

    def save_model(self, path: str):
        """モデルを保存します。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Pickleでラッパーごと保存するか、model.save_modelを使うか
        # ここでは統一性のためPickleを使用
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"モデルを保存しました: {path}")

    def load_model(self, path: str):
        """モデルをロードします。"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"モデルをロードしました: {path}")

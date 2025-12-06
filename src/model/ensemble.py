import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LinearRegression
import pickle
import os

from model.lgbm import KeibaLGBM
from model.catboost_model import KeibaCatBoost
from model.tabnet_model import KeibaTabNet

logger = logging.getLogger(__name__)

class EnsembleModel:
    """
    LightGBM, CatBoost, TabNetのアンサンブル（Stacking/Blending）モデル。
    """
    def __init__(self):
        self.lgbm = KeibaLGBM()
        self.catboost = KeibaCatBoost()
        self.tabnet = KeibaTabNet()
        self.meta_model = LinearRegression() # シンプルな線形回帰で重み付け

    def train(self, train_set: dict, valid_set: dict):
        """
        Level 1モデルとMetaモデルを学習します。
        今回はValidセットを使ってMetaモデルを学習するBlending方式を採用します。
        """
        logger.info("アンサンブル学習開始...")

        # 1. LightGBM学習
        logger.info("--- LightGBM ---")
        self.lgbm.train(train_set, valid_set)

        # 2. CatBoost学習
        logger.info("--- CatBoost ---")
        self.catboost.train(train_set, valid_set)

        # 3. TabNet学習
        logger.info("--- TabNet ---")
        self.tabnet.train(train_set, valid_set)

        # 4. メタ特徴量の作成 (Validセットに対する予測値)
        logger.info("--- Meta Model ---")
        pred_lgbm = self.lgbm.predict(valid_set['X'])
        pred_cb = self.catboost.predict(valid_set['X'])
        pred_tab = self.tabnet.predict(valid_set['X'])

        X_meta = pd.DataFrame({
            'lgbm': pred_lgbm,
            'cb': pred_cb,
            'tabnet': pred_tab
        })
        y_meta = valid_set['y'] # ターゲット (Relevance Score)

        # 5. Metaモデル学習
        self.meta_model.fit(X_meta, y_meta)

        weights = self.meta_model.coef_
        intercept = self.meta_model.intercept_
        logger.info(f"Meta Model Weights: LGBM={weights[0]:.4f}, CatBoost={weights[1]:.4f}, TabNet={weights[2]:.4f}, Bias={intercept:.4f}")
        logger.info("アンサンブル学習完了")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p1 = self.lgbm.predict(X)
        p2 = self.catboost.predict(X)
        p3 = self.tabnet.predict(X)
        X_meta = pd.DataFrame({
            'lgbm': p1,
            'cb': p2,
            'tabnet': p3
        })
        return self.meta_model.predict(X_meta)

    def save_model(self, path: str):
        # EnsembleModel自体をpickleするが、TabNetは別ファイル管理が必要なため、
        # TabNet以外の部分と、TabNetのパス管理を行う必要がある。

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # TabNet保存
        tabnet_path = path.replace('.pkl', '_tabnet.pkl')
        self.tabnet.save_model(tabnet_path)

        # 一時的にTabNetを外して保存（Pickleできない可能性があるため）
        temp_tabnet = self.tabnet
        self.tabnet = None

        with open(path, 'wb') as f:
            pickle.dump(self, f)

        # 復元
        self.tabnet = temp_tabnet
        logger.info(f"アンサンブルモデルを保存しました: {path}")

    def load_model(self, path: str):
        # メインのロード
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        self.lgbm = loaded.lgbm
        self.catboost = loaded.catboost
        self.meta_model = loaded.meta_model

        # TabNetのロード
        self.tabnet = KeibaTabNet()
        tabnet_path = path.replace('.pkl', '_tabnet.pkl')
        if os.path.exists(tabnet_path.replace('.pkl', '.zip')): # TabNet saves as zip
             self.tabnet.load_model(tabnet_path)
        else:
             logger.warning(f"TabNetモデルが見つかりません: {tabnet_path}")

        logger.info(f"アンサンブルモデルをロードしました: {path}")

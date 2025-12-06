import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LinearRegression
import pickle
import os

# srcディレクトリをパスに追加していなくても、このモジュールがimportされるときは
# sys.pathが設定されている前提、あるいは相対importを使う
# ここでは絶対importを使うため、呼び出し元でパス設定が必要
from model.lgbm import KeibaLGBM
from model.catboost_model import KeibaCatBoost

logger = logging.getLogger(__name__)

class EnsembleModel:
    """
    LightGBMとCatBoostのアンサンブル（Stacking/Blending）モデル。
    """
    def __init__(self):
        self.lgbm = KeibaLGBM()
        self.catboost = KeibaCatBoost()
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

        # 3. メタ特徴量の作成 (Validセットに対する予測値)
        logger.info("--- Meta Model ---")
        pred_lgbm = self.lgbm.predict(valid_set['X'])
        pred_cb = self.catboost.predict(valid_set['X'])

        X_meta = pd.DataFrame({'lgbm': pred_lgbm, 'cb': pred_cb})
        y_meta = valid_set['y'] # ターゲット (Relevance Score)

        # 4. Metaモデル学習
        self.meta_model.fit(X_meta, y_meta)

        weights = self.meta_model.coef_
        intercept = self.meta_model.intercept_
        logger.info(f"Meta Model Weights: LGBM={weights[0]:.4f}, CatBoost={weights[1]:.4f}, Bias={intercept:.4f}")
        logger.info("アンサンブル学習完了")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p1 = self.lgbm.predict(X)
        p2 = self.catboost.predict(X)
        X_meta = pd.DataFrame({'lgbm': p1, 'cb': p2})
        return self.meta_model.predict(X_meta)

    def save_model(self, path: str):
        # EnsembleModel自体をpickleするが、TabNetは別ファイル管理が必要なため、
        # TabNet以外の部分と、TabNetのパス管理を行う必要がある。

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"アンサンブルモデルを保存しました: {path}")

    def load_model(self, path: str):
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        self.lgbm = loaded.lgbm
        self.catboost = loaded.catboost
        self.meta_model = loaded.meta_model
        logger.info(f"アンサンブルモデルをロードしました: {path}")

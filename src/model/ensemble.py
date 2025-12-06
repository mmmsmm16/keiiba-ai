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

    def load_base_models(self, model_dir: str):
        """
        学習済みのBase Models (LightGBM, CatBoost, TabNet) をロードします。
        """
        logger.info(f"Base Modelsをロードします: {model_dir}")
        
        # LightGBM
        lgbm_path = os.path.join(model_dir, 'lgbm.pkl')
        if os.path.exists(lgbm_path):
            self.lgbm.load_model(lgbm_path)
        else:
            logger.warning(f"LightGBMモデルが見つかりません: {lgbm_path}")

        # CatBoost
        catboost_path = os.path.join(model_dir, 'catboost.pkl')
        if os.path.exists(catboost_path):
            self.catboost.load_model(catboost_path)
        else:
            logger.warning(f"CatBoostモデルが見つかりません: {catboost_path}")

        # TabNet
        tabnet_path = os.path.join(model_dir, 'tabnet.zip')
        if os.path.exists(tabnet_path):
            self.tabnet.load_model(tabnet_path.replace('.zip', '')) # load_model adds .zip internally or handles it
        else:
            logger.warning(f"TabNetモデルが見つかりません: {tabnet_path}")

    def train_meta_model(self, valid_set: dict):
        """
        Base Modelsの予測値を使ってMeta Model (LinearRegression) を学習します。
        注意: ここでは train_set ではなく valid_set を使用して Blending を行います。
        (本来はOut-of-Fold予測を使うべきですが、簡易化のためValid使用)
        """
        logger.info("Meta Modelの学習を開始します...")
        
        # 予測値の取得
        logger.info("予測値の生成中...")
        pred_lgbm = self.lgbm.predict(valid_set['X'])
        pred_cb = self.catboost.predict(valid_set['X'])
        pred_tab = self.tabnet.predict(valid_set['X'])

        X_meta = pd.DataFrame({
            'lgbm': pred_lgbm,
            'cb': pred_cb,
            'tabnet': pred_tab
        })
        y_meta = valid_set['y']

        # Metaモデル学習
        self.meta_model.fit(X_meta, y_meta)

        weights = self.meta_model.coef_
        intercept = self.meta_model.intercept_
        logger.info(f"Meta Model Weights: LGBM={weights[0]:.4f}, CatBoost={weights[1]:.4f}, TabNet={weights[2]:.4f}, Bias={intercept:.4f}")
        logger.info("Meta Modelの学習が完了しました。")

    def train(self, train_set: dict, valid_set: dict):
        """
        [Deprecated] 一括学習用メソッド。将来的に削除予定。
        """
        logger.info("アンサンブル一括学習開始 (非推奨)...")
        self.lgbm.train(train_set, valid_set)
        self.catboost.train(train_set, valid_set)
        self.tabnet.train(train_set, valid_set)
        self.train_meta_model(valid_set)
        logger.info("アンサンブル一括学習完了")

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

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
        # GPU競合 (CUBLAS Error) が解決したと思われるため、GPU利用を許可する (default: auto detect)
        self.tabnet = KeibaTabNet()

        self.meta_model = LinearRegression() # シンプルな線形回帰で重み付け
        
        # フラグ初期化
        self.has_lgbm = False
        self.has_catboost = False
        self.has_tabnet = False

    def load_base_models(self, model_dir: str, version: str = 'v1'):
        """
        学習済みのBase Models (LightGBM, CatBoost, TabNet) をロードします。
        Args:
            model_dir (str): モデルディレクトリのパス
            version (str): ロードするモデルのバージョン (デフォルト: v1)
        """
        logger.info(f"Base Models (Version: {version}) をロードします: {model_dir}")
        self.version = version
        
        # LightGBM
        lgbm_path = os.path.join(model_dir, f'lgbm_{version}.pkl')
        if os.path.exists(lgbm_path):
            self.lgbm.load_model(lgbm_path)
            self.has_lgbm = True
        else:
            logger.warning(f"LightGBMモデルが見つかりません: {lgbm_path}")
            self.has_lgbm = False

        # CatBoost
        catboost_path = os.path.join(model_dir, f'catboost_{version}.pkl')
        if os.path.exists(catboost_path):
            self.catboost.load_model(catboost_path)
            self.has_catboost = True
        else:
            logger.warning(f"CatBoostモデルが見つかりません: {catboost_path}")
            self.has_catboost = False

        # TabNet
        # v7ではTabNet最適化を行っていないためスキップされる可能性がある
        tabnet_path = os.path.join(model_dir, f'tabnet_{version}.zip')
        if os.path.exists(tabnet_path):
            self.tabnet.load_model(tabnet_path.replace('.zip', ''), device_name='cpu')
            self.has_tabnet = True
        else:
            logger.info(f"TabNetモデルが見つかりません (スキップします): {tabnet_path}")
            self.has_tabnet = False

    def train_meta_model(self, valid_set: dict):
        """
        Base Modelsの予測値を使ってMeta Model (LinearRegression) を学習します。
        注意: ここでは train_set ではなく valid_set を使用して Blending を行います。
        (本来はOut-of-Fold予測を使うべきですが、簡易化のためValid使用)
        """
        logger.info("Meta Modelの学習を開始します...")
        
        # 予測値の取得
        logger.info("予測値の生成中...")
        preds = {}
        if self.has_lgbm:
            preds['lgbm'] = self.lgbm.predict(valid_set['X'])
        if self.has_catboost:
            preds['cb'] = self.catboost.predict(valid_set['X'])
        if self.has_tabnet:
            preds['tabnet'] = self.tabnet.predict(valid_set['X'])
            
        if not preds:
            raise ValueError("有効なBase Modelが1つもありません。")

        X_meta = pd.DataFrame(preds)
        y_meta = valid_set['y']

        # Metaモデル学習
        self.meta_model.fit(X_meta, y_meta)

        weights = self.meta_model.coef_
        intercept = self.meta_model.intercept_
        
        weight_str = ", ".join([f"{k}={w:.4f}" for k, w in zip(X_meta.columns, weights)])
        logger.info(f"Meta Model Weights: {weight_str}, Bias={intercept:.4f}")
        logger.info("Meta Modelの学習が完了しました。")

    def train(self, train_set: dict, valid_set: dict):
        """
        [Deprecated] 一括学習用メソッド。将来的に削除予定。
        """
        logger.info("アンサンブル一括学習開始 (非推奨)...")
        # 個別モデルの学習は完了している前提
        self.train_meta_model(valid_set)
        logger.info("アンサンブル一括学習完了")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = {}
        if self.has_lgbm:
            preds['lgbm'] = self.lgbm.predict(X)
        if self.has_catboost:
            preds['cb'] = self.catboost.predict(X)
        if self.has_tabnet:
            preds['tabnet'] = self.tabnet.predict(X)
            
        if not preds:
            # フォールバック: 全て0
            return np.zeros(len(X))

        X_meta = pd.DataFrame(preds)
        return self.meta_model.predict(X_meta)

    def save_model(self, path: str):
        # EnsembleModel自体をpickleするが、TabNetは別ファイル管理が必要なため、
        # TabNet以外の部分と、TabNetのパス管理を行う必要がある。

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # TabNet保存 (存在する場合のみ)
        temp_tabnet = self.tabnet
        if hasattr(self, 'has_tabnet') and self.has_tabnet:
            tabnet_path = path.replace('.pkl', '_tabnet.pkl')
            self.tabnet.save_model(tabnet_path)
            # 一時的にTabNetを外して保存（Pickleできない可能性があるため）
            self.tabnet = None
        else:
            # TabNetがない場合はNoneにしておく
            self.tabnet = None

        with open(path, 'wb') as f:
            pickle.dump(self, f)

        # 復元
        self.tabnet = temp_tabnet
        logger.info(f"アンサンブルモデルを保存しました: {path}")

    def load_model(self, path: str, device_name: str = None):
        # メインのロード
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        self.lgbm = loaded.lgbm
        self.catboost = loaded.catboost
        self.meta_model = loaded.meta_model
        
        # 保存時にhas_lgbm等が保存されているはずだが、念のため復元
        # 古いモデルファイルで属性がない場合は、存在チェックを行うか、Trueとみなす
        self.has_lgbm = getattr(loaded, 'has_lgbm', True)
        self.has_catboost = getattr(loaded, 'has_catboost', True)
        self.has_tabnet = getattr(loaded, 'has_tabnet', False) # TabNetは以前はなかったのでFalseデフォルトが安全かも?でもv3はあった?
        # v3まではTabNetはEnsembleに含まれてたか？ -> Ensembleのpredictで使われたならTrueにすべき。
        # しかし古いpickleにTabNetが含まれてなければエラーになるため、
        # TabNetについては安全側に倒してFalseにするか、loadedからtabnet属性があるか見る。
        
        # 属性としての存在確認 (Noneでないか)
        if self.lgbm is None: self.has_lgbm = False
        if self.catboost is None: self.has_catboost = False

        # TabNetのロード (GPU利用可, 推論時はCPU強制も可)
        self.tabnet = KeibaTabNet()
        if hasattr(self, 'has_tabnet') and self.has_tabnet:
            tabnet_path = path.replace('.pkl', '_tabnet.pkl')
            # zip拡張子の確認
            if os.path.exists(tabnet_path.replace('.pkl', '.zip')) or os.path.exists(tabnet_path + '.zip'): 
                 self.tabnet.load_model(tabnet_path, device_name=device_name)
            else:
                 logger.warning(f"TabNetモデルが見つかりません: {tabnet_path}")
                 self.has_tabnet = False # ロード失敗したらFalseにする


        logger.info(f"アンサンブルモデルをロードしました: {path}")

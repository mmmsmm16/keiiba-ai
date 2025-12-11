import pandas as pd
import numpy as np
import logging
import os
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt

import json

logger = logging.getLogger(__name__)

class KeibaTabNet:
    """
    TabNetを使用した競馬予測モデルクラス。
    Rankingタスクとして扱うため、回帰モデル（Regressor）を使用して着順スコア（逆数や正規化ランク）を学習するか、
    あるいは単純に着順そのものを学習させます。

    今回は、他のモデル(LGBM LambdaRank)とのアンサンブルを考慮し、
    出力が「高いほど良い」スコアになるようにターゲットを変換して学習させます。
    (例: log(1/rank) や 単純な正規化)
    """
    def __init__(self, params=None):
        self.params = params or {}

        # デフォルトパラメータ
        default_params = {
            'n_d': 32, # Width of the decision prediction layer
            'n_a': 32, # Width of the attention embedding for each step
            'n_steps': 5, # Number of steps in the architecture
            'gamma': 1.3, # Coefficient for feature reusage in the masks
            'n_independent': 2, # Number of independent Gated Linear Units layers at each step
            'n_shared': 2, # Number of shared Gated Linear Units layers at each step
            'lambda_sparse': 1e-3, # Sparsity loss coefficient
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': dict(lr=2e-2),
            'scheduler_params': dict(step_size=50, gamma=0.9),
            'scheduler_fn': torch.optim.lr_scheduler.StepLR,
            'mask_type': 'entmax', # "sparsemax", "entmax"
            'verbose': 1,
            'device_name': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

        # fitメソッド用のパラメータを分離
        self.fit_params = {
            'max_epochs': self.params.pop('max_epochs', 100),
            'patience': self.params.pop('patience', 20),
            'batch_size': self.params.pop('batch_size', 512),
            'virtual_batch_size': self.params.pop('virtual_batch_size', 128),
            'num_workers': self.params.pop('num_workers', 0),
            'drop_last': self.params.pop('drop_last', False)
        }

        # 残りのパラメータをモデル初期化用にマージ
        init_params = default_params.copy()
        init_params.update(self.params)

        self.params = init_params # self.paramsを更新して保持

        # TabNetRegressorのinitに渡さないパラメータを除去
        self.params.pop('enabled', None)

        self.model = TabNetRegressor(**self.params)
        self.scaler = StandardScaler()
        self.fitted_scaler = False
        self.feature_names = None

    def _preprocess_target(self, y):
        """
        LightGBM(LambdaRank)などは大きい値ほど良いランクとするため、
        TabNetでも同様の傾向を持つターゲットに変換して学習させる。
        例: 1着 -> 1.0, 18着 -> 0.0 のようなスコア、あるいは log(1/rank)

        ここではシンプルに逆数変換を用いる: 1/rank
        1着=1.0, 2着=0.5, ... 18着=0.05
        """
        # yがSeriesの場合、valuesを取得
        if isinstance(y, pd.Series):
            y = y.values

        # 入力 y は DatasetSplitter で作成された "Relevance Score" (3=1着, 2=2着, 1=3着, 0=着外)
        # つまり「高いほど良い」値になっている。
        # そのため、逆数変換などの「低いほど良い→高いほど良い」への変換は【不要】であり、むしろ有害（逆転してしまう）。
        # よって、そのまま float に変換して返す。
        return y.astype(float)

    def train(self, train_set: dict, valid_set: dict):
        """
        モデルを学習させます。
        TabNetはPandas DataFrameではなくNumPy arrayを期待します。
        """
        logger.info(f"TabNetの学習を開始します... Device: {self.params.get('device_name')}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 特徴量名を保持
        logger.info("Step 1/5: 特徴量名の保持...")
        if isinstance(train_set['X'], pd.DataFrame):
            self.feature_names = train_set['X'].columns.tolist()
            logger.info(f"  特徴量数: {len(self.feature_names)}, Train行数: {len(train_set['X'])}")
            X_train_df = train_set['X'].copy()
            X_valid_df = valid_set['X'].copy()
            
            # カテゴリ/オブジェクト型の処理 (簡易的にコード化)
            cat_cols = X_train_df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                logger.info(f"Step 2/5: カテゴリ変数の変換 ({len(cat_cols)}列)...")
                for col in cat_cols:
                    logger.info(f"  Converting column '{col}' to numeric codes.")
                    combined = pd.concat([X_train_df[col], X_valid_df[col]], axis=0).astype('category')
                    X_train_df[col] = combined.iloc[:len(X_train_df)].cat.codes
                    X_valid_df[col] = combined.iloc[len(X_train_df):].cat.codes
            else:
                logger.info("Step 2/5: カテゴリ変数なし（スキップ）")

            logger.info("Step 3/5: NumPy配列への変換...")
            X_train = X_train_df.values
            X_valid = X_valid_df.values
        else:
            X_train = train_set['X']
            X_valid = valid_set['X']

        y_train_raw = train_set['y']
        y_valid_raw = valid_set['y']

        # ターゲット変換
        logger.info("Step 4/5: ターゲット変換とfloat32キャスト...")
        y_train = self._preprocess_target(y_train_raw).reshape(-1, 1).astype(np.float32)
        y_valid = self._preprocess_target(y_valid_raw).reshape(-1, 1).astype(np.float32)

        # 特徴量のスケーリング
        try:
            X_train = np.array(X_train, dtype=np.float32)
            X_valid = np.array(X_valid, dtype=np.float32)
        except Exception as e:
            logger.error(f"TabNet: Failed to convert data to float32. Check for non-numeric values. Error: {e}")
            raise e

        logger.info("Step 5/5: スケーリングとNaN処理...")
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_valid = np.nan_to_num(X_valid, nan=0.0)

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_valid_scaled = self.scaler.transform(X_valid)
        self.fitted_scaler = True
        
        max_epochs = self.fit_params['max_epochs']
        logger.info(f"データ準備完了。TabNet学習を開始します... (max_epochs={max_epochs})")

        # プログレスバー用カスタムコールバック
        from pytorch_tabnet.callbacks import Callback
        
        class EpochProgressCallback(Callback):
            def __init__(self, max_epochs):
                super().__init__()
                self.max_epochs = max_epochs
                
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                train_loss = logs.get('loss', 0)
                valid_rmse = logs.get('valid_rmse', 0)
                print(f"Epoch {epoch+1}/{self.max_epochs} | loss: {train_loss:.5f} | valid_rmse: {valid_rmse:.5f}", flush=True)

        self.model.fit(
            X_train=X_train_scaled,
            y_train=y_train,
            eval_set=[(X_train_scaled, y_train), (X_valid_scaled, y_valid)],
            eval_name=['train', 'valid'],
            eval_metric=['rmse', 'mae'],
            max_epochs=max_epochs,
            patience=self.fit_params['patience'],
            batch_size=self.fit_params['batch_size'],
            virtual_batch_size=self.fit_params['virtual_batch_size'],
            num_workers=self.fit_params['num_workers'],
            drop_last=self.fit_params['drop_last'],
            callbacks=[EpochProgressCallback(max_epochs)]
        )
        logger.info("学習が完了しました。")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        推論を行います。
        """
        if not self.fitted_scaler:
             raise ValueError("モデル（スケーラー）が学習されていません。")

        # Just-In-Time Device Fix (for Streamlit Caching issues)
        if hasattr(self, 'model') and hasattr(self.model, 'network'):
             # Check if we expect CPU
             if getattr(self.model, 'device_name', '') == 'cpu':
                  # Check explicit params or buffers
                  # group_attention_matrix is inside encoder
                  if hasattr(self.model.network, 'tabnet'):
                       tabnet_module = self.model.network.tabnet
                       # It might be directly in tabnet or in encoder
                       modules_to_check = [tabnet_module]
                       if hasattr(tabnet_module, 'encoder'):
                           modules_to_check.append(tabnet_module.encoder)
                       
                       for mod in modules_to_check:
                           if hasattr(mod, 'group_attention_matrix'):
                               buf = mod.group_attention_matrix
                               if buf.device.type != 'cpu':
                                   logger.warning(f"JIT Fix: Moving group_attention_matrix from {buf.device} to cpu")
                                   mod.group_attention_matrix = buf.cpu()
                  
                  # Ensure whole network is on CPU
                  # accessing first param to check
                  try:
                      p = next(self.model.network.parameters())
                      if p.device.type != 'cpu':
                          logger.warning(f"JIT Fix: Moving whole network from {p.device} to cpu")
                          self.model.network.cpu()
                  except: pass

        # 特徴量数の整合性チェックとフィルタリング
        if hasattr(self.scaler, 'n_features_in_'):
            expected_features = self.scaler.n_features_in_
            if X.shape[1] != expected_features:
                if self.feature_names and len(self.feature_names) == expected_features:
                    # 特徴量名が保存されている場合は名前でフィルタリング
                    missing = set(self.feature_names) - set(X.columns)
                    if not missing:
                        X = X[self.feature_names]
                elif X.shape[1] > expected_features:
                    # 名前がない場合（旧モデル）は、先頭から必要数だけスライスする（ヒューリスティック）
                    # 新しい特徴量は末尾に追加される傾向があるため
                    logger.warning(f"Feature count mismatch (Expected: {expected_features}, Got: {X.shape[1]}). Slicing first {expected_features} columns.")
                    X = X.iloc[:, :expected_features]
                else:
                    raise ValueError(f"Feature count mismatch (Expected: {expected_features}, Got: {X.shape[1]}) and simpler slicing impossible.")

        # カテゴリ変数を数値に変換 (train()と同様)
        X_converted = X.copy()
        for col in X_converted.columns:
            if X_converted[col].dtype == 'object' or X_converted[col].dtype.name == 'category':
                X_converted[col] = X_converted[col].astype('category').cat.codes

        X_values = X_converted.values.astype(np.float32)
        X_values = np.nan_to_num(X_values, nan=0.0)
        X_scaled = self.scaler.transform(X_values)

        # TabNetRegressor returns shape (N, 1)
        # Ensure input is float32 (torch requirement usually)
        # self.model.predict handles numpy conversion, but ensures device alignment
        preds = self.model.predict(X_scaled)
        return preds.flatten()

    def save_model(self, path: str):
        """モデルとスケーラーを保存します。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # モデル本体 (.zip拡張子はTabNetが自動追加するので除去)
        model_path = path.replace('.pkl', '').replace('.zip', '')
        self.model.save_model(model_path) # saves as model_path.zip

        # スケーラー
        scaler_path = model_path + '.scaler'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
        # 特徴量名 (あれば保存)
        if self.feature_names:
            feature_path = model_path + '.features.json'
            with open(feature_path, 'w') as f:
                json.dump(self.feature_names, f)

        logger.info(f"TabNetモデルを保存しました: {model_path}.zip")

    def load_model(self, path: str, device_name: str = None):
        """モデルとスケーラーをロードします。"""
        # モデル本体 (.zip拡張子を正しく処理)
        base_path = path.replace('.pkl', '').replace('.zip', '')
        model_path = base_path + '.zip'
        self.model.load_model(model_path)

        # デバイスの強制変更 (Inference用)
        if device_name:
            self.model.device_name = device_name
            # Explicitly set device to ensure predict uses correct device for tensors
            self.model.device = torch.device(device_name)
            
            if device_name == 'cpu':
                self.model.preds_mapper = {k: v.decode("utf-8") for k, v in self.model.preds_mapper.items()} if hasattr(self.model, 'preds_mapper') and self.model.preds_mapper else {}
            
            # Move network to device
            if hasattr(self.model, 'network'):
                self.model.network.to(torch.device(device_name))
                param_device = next(self.model.network.parameters()).device
                logger.debug(f"TabNet Device confirmed: model.device={self.model.device}, network.param={param_device}")

                # Verify group_attention_matrix specifically (buffer)
                if hasattr(self.model.network, 'tabnet') and hasattr(self.model.network.tabnet, 'group_attention_matrix'):
                     buf = self.model.network.tabnet.group_attention_matrix
                     logger.debug(f"DEBUG: Group Attention Matrix Device: {buf.device}")
                     if device_name == 'cpu' and buf.device.type != 'cpu':
                         logger.warning("Force moving group_attention_matrix to CPU")
                         self.model.network.tabnet.group_attention_matrix = buf.cpu()

            logger.info(f"TabNet Device set to: {device_name}")

        # スケーラー
        scaler_path = base_path + '.scaler'
        if not os.path.exists(scaler_path):
             scaler_path = path + '.scaler'  # 古い形式のフォールバック
        if not os.path.exists(scaler_path):
             scaler_path = path + '.pkl.scaler'  # さらに古い形式

        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.fitted_scaler = True
            logger.info(f"スケーラーをロードしました: {scaler_path}")
        else:
            logger.warning(f"スケーラーファイルが見つかりません: {scaler_path}")
            # TabNetはスケール変換必須のため、ここでWarning出すだけだと後でエラーになるが、
            # 訓練済みモデルがないケースもありうるのでExceptionにはしないでおく
            # ただし predict 時にはチェックされる

        # 特徴量名
        feature_path = path + '.features.json'
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                self.feature_names = json.load(f)
            logger.info(f"特徴量名をロードしました: {len(self.feature_names)} features")
        else:
            self.feature_names = None
            logger.info("特徴量名ファイルが見つかりません (旧モデル互換モードで動作します)")

        logger.info(f"TabNetモデルをロードしました: {model_path}")

    def plot_importance(self, output_path: str):
        """特徴量重要度をプロットして保存します。"""
        try:
            feat_importances = self.model.feature_importances_
            indices = np.argsort(feat_importances)[::-1]
            indices = indices[:20]

            plt.figure(figsize=(10, 6))
            plt.title("Feature Importances (TabNet)")
            plt.bar(range(len(indices)), feat_importances[indices], align="center")
            plt.xticks(range(len(indices)), indices)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            logger.info(f"特徴量重要度を保存しました: {output_path}")
        except Exception as e:
            logger.warning(f"特徴量重要度のプロットに失敗しました: {e}")

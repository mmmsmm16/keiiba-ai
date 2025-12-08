
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class FeatureAdapter:
    """
    モデルが必要とする特徴量と入力データフレームを整合させるアダプター。
    新しいバージョンで追加された特徴量を削除したり、
    古いバージョンで必要だったが削除された特徴量を補完したりします。
    """
    def __init__(self, model):
        self.model = model
        self.expected_features = self._get_expected_features(model)
        
    def _get_expected_features(self, model):
        """
        モデルオブジェクトから期待される特徴量リストを抽出します。
        サポート: LightGBM (Booster), CatBoost, TabNet (要調査)
        """
        try:
            # LightGBM (Booster)
            if hasattr(model, 'feature_name'):
                return model.feature_name()
            
            # CatBoost
            if hasattr(model, 'feature_names_'):
                return model.feature_names_
            
            # Sklearn-like / TabNet
            if hasattr(model, 'feature_names_in_'):
                return list(model.feature_names_in_)
                
            logger.warning("モデルから特徴量リストを取得できませんでした。アダプターは機能しません。")
            return None
        except Exception as e:
            logger.warning(f"特徴量リスト取得中にエラー: {e}")
            return None

    def adapt(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データフレームをモデルの入力形式に合わせます。
        """
        if self.expected_features is None:
            return df
            
        current_cols = set(df.columns)
        expected_cols = self.expected_features
        
        # 1. 不足しているカラムを補完 (Missing Columns) -> 0埋め or NaN
        # 古いモデルが必要としていたが、現在は生成されていない特徴量など
        missing = [c for c in expected_cols if c not in current_cols]
        if missing:
            logger.info(f"不足特徴量を補完します (0埋め): {len(missing)}個 (例: {missing[:3]})")
            for c in missing:
                df[c] = 0.0 # とりあえず0で埋める（影響を最小限に）
        
        # 2. 不要なカラムを削除 & 並び順を統一
        # モデルが期待する順序で返すことが重要 (特にLightGBM/CatBoostの一部)
        df_adapted = df[expected_cols].copy()
        
        return df_adapted

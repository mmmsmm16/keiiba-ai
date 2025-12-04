import pandas as pd
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    基本特徴量の生成を行うクラス。
    """
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        カテゴリ変数の数値化や日付特徴量の追加を行います。

        Args:
            df (pd.DataFrame): クレンジング済みのデータ。

        Returns:
            pd.DataFrame: 特徴量が追加されたデータ。
        """
        logger.info("基本特徴量の生成を開始...")

        # 1. 性別の数値化
        # 牡->1, 牝->2, セ->3
        sex_map = {'牡': 1, '牝': 2, 'セ': 3}
        df['sex_num'] = df['sex'].map(sex_map).fillna(0).astype(int)

        # 2. 天候の数値化
        # 晴->1, 曇->2, 雨->3, 小雨->4, 雪->5
        weather_map = {'晴': 1, '曇': 2, '雨': 3, '小雨': 4, '雪': 5}
        # マップにないものは0 (Unknown)
        df['weather_num'] = df['weather'].map(weather_map).fillna(0).astype(int)

        # 3. 馬場種別の数値化
        # 芝->1, ダート->2, 障害->3
        surface_map = {'芝': 1, 'ダート': 2, '障害': 3}
        df['surface_num'] = df['surface'].map(surface_map).fillna(0).astype(int)

        # 4. 馬場状態の数値化
        # 良->1, 稍重->2, 重->3, 不良->4
        state_map = {'良': 1, '稍重': 2, '重': 3, '不良': 4}
        df['state_num'] = df['state'].map(state_map).fillna(0).astype(int)

        # 5. 日付情報の抽出
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.weekday # 0:Monday, 6:Sunday

        logger.info("基本特徴量の生成完了")
        return df

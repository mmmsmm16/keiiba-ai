import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataCleanser:
    """
    生データのクレンジングを行うクラス。
    """
    def cleanse(self, df: pd.DataFrame, is_inference: bool = False) -> pd.DataFrame:
        """
        データフレームのクレンジングを行います。

        Args:
            df (pd.DataFrame): RawDataLoaderから取得した生データ。
            is_inference (bool): 推論用データかどうか。Trueの場合、着順(rank)がなくても削除しません。

        Returns:
            pd.DataFrame: クレンジング済みのデータ。
        """
        logger.info(f"データクレンジングを開始... (Inference Mode: {is_inference})")
        original_len = len(df)

        # 1. 順位(rank)がないデータの処理
        if not is_inference:
            # 学習時は、順位がない（取消・中止など）データは削除
            df = df.dropna(subset=['rank'])
            # 順位を整数型に変換
            df['rank'] = df['rank'].astype(int)
        else:
            # 推論時は、rankがNaNでも保持する (未来のレースなので当然NaN)
            # ただし、rankカラム自体は存在することを保証（NaNのまま）
            if 'rank' not in df.columns:
                df['rank'] = float('nan')

        # 2. 欠損値の処理
        # 体重(weight)の欠損: 平均値で埋める
        if df['weight'].isnull().any():
            mean_weight = df['weight'].mean()
            df['weight'] = df['weight'].fillna(mean_weight)

        # 体重増減(weight_diff): 0で埋める
        df['weight_diff'] = df['weight_diff'].fillna(0)

        # 3. 型変換
        # 日付をdatetime型に
        df['date'] = pd.to_datetime(df['date'])

        logger.info(f"クレンジング完了: {original_len} -> {len(df)} 件 (削除: {original_len - len(df)})")
        return df

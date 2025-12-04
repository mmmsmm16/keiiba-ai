import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataCleanser:
    """
    生データのクレンジングを行うクラス。
    """
    def cleanse(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データフレームのクレンジングを行います。

        Args:
            df (pd.DataFrame): RawDataLoaderから取得した生データ。

        Returns:
            pd.DataFrame: クレンジング済みのデータ。
        """
        logger.info("データクレンジングを開始...")
        original_len = len(df)

        # 1. 順位(rank)がないデータ（取消、中止など）を削除
        df = df.dropna(subset=['rank'])
        # 順位を整数型に変換
        df['rank'] = df['rank'].astype(int)

        # 2. 欠損値の処理
        # 体重(weight)の欠損: 平均値で埋める (あるいは前走参照だが、ここでは簡易的に平均)
        # 本格的にはLeakageに注意が必要だが、体重は事前にわかる情報なので全体の平均で埋めても大きな問題ではない（厳密には開催日以前の平均であるべき）
        # 簡易MVPとして、そのレース内の平均、あるいは全体平均で埋める。
        if df['weight'].isnull().any():
            mean_weight = df['weight'].mean()
            df['weight'] = df['weight'].fillna(mean_weight)

        # 体重増減(weight_diff): 0で埋める
        df['weight_diff'] = df['weight_diff'].fillna(0)

        # 上がり3F(last_3f): 欠損の場合は平均で埋める（レース結果なので学習には使わないが、念のため）
        # ただし、last_3fは予測対象ではなく特徴量（過去走）として使う場合、ここでの欠損補完は重要。
        # 今は「現在のレース結果」のクレンジングなので、そのままにしておくか、適当な値を入れる。

        # 3. 型変換
        # 日付をdatetime型に
        df['date'] = pd.to_datetime(df['date'])

        logger.info(f"クレンジング完了: {original_len} -> {len(df)} 件 (削除: {original_len - len(df)})")
        return df

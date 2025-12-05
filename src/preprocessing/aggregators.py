import pandas as pd
import logging

logger = logging.getLogger(__name__)

class HistoryAggregator:
    """
    馬の過去走情報（ラグ特徴量）を生成するクラス。
    """
    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        過去走データを集計して特徴量を追加します。

        Args:
            df (pd.DataFrame): 前処理済みの全データ。

        Returns:
            pd.DataFrame: 過去走特徴量が追加されたデータ。
        """
        logger.info("過去走特徴量の生成を開始...")

        # 処理のために馬IDと日付でソート（必須）
        df = df.sort_values(['horse_id', 'date'])

        # 特徴量生成対象のカラム
        target_cols = ['rank', 'last_3f', 'odds', 'popularity']

        # 1. 前走（Lag 1）の特徴量
        # 単純に1つずらす
        logger.info("前走データの生成中...")
        grouped = df.groupby('horse_id')
        for col in target_cols:
            df[f'lag1_{col}'] = grouped[col].shift(1)

        # 2. 近5走の平均（Rolling Mean）
        # 現在のレースを含まないように、shift(1)した上でrollingする
        logger.info("近5走平均データの生成中...")
        for col in ['rank', 'last_3f']:
            # transformを使うことで元のインデックスを維持
            df[f'mean_{col}_5'] = grouped[col].transform(lambda x: x.shift(1).rolling(5).mean())

        # 3. コース適性（同馬場状態・同距離区分での過去成績）
        # 実装簡略化のため、今回は「場所(Venue)」ではなく「馬場種別(Surface)」ごとの通算平均着順を計算
        # Expanding mean (累積平均) を使用
        # これも現在のレースを含んでしまうとリークになるので、shift(1)が必要

        # Surfaceごとのグルーピングは少し複雑になるため、ここでは「通算平均着順」を計算
        logger.info("通算成績データの生成中...")
        df['total_races'] = grouped['race_id'].cumcount() # 過去出走数

        # 過去の平均着順 (Expanding Mean)
        # shift(1)してからexpanding
        df['mean_rank_all'] = grouped['rank'].transform(lambda x: x.shift(1).expanding().mean())

        # 過去の勝利数
        # rank==1 を 1, その他 0 に変換して sum
        df['wins_all'] = grouped['rank'].transform(lambda x: (x == 1).astype(int).shift(1).expanding().sum())

        # 勝率
        df['win_rate_all'] = df['wins_all'] / df['total_races']
        # 0除算対策（total_racesが0の場合はNaNになるが、0で埋める）
        df['win_rate_all'] = df['win_rate_all'].fillna(0)

        logger.info("過去走特徴量の生成完了")
        return df

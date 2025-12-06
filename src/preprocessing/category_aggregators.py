import pandas as pd
import logging

logger = logging.getLogger(__name__)

class CategoryAggregator:
    """
    カテゴリ変数（騎手、調教師、種牡馬など）の過去成績を集計するクラス。
    ターゲットエンコーディングに近いですが、リークを防ぐために過去データのみを使用します。
    """
    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        カテゴリごとの集計特徴量（勝率、複勝率、出走回数）を追加します。

        Args:
            df (pd.DataFrame): 前処理済みのデータ。

        Returns:
            pd.DataFrame: 集計特徴量が追加されたデータ。
        """
        logger.info("カテゴリ集計特徴量の生成を開始...")

        # 時系列順であることを保証（GroupBy後の順序維持のため）
        df = df.sort_values(['date', 'race_id'])

        # 集計対象のカテゴリ列
        targets = ['jockey_id', 'trainer_id', 'sire_id']

        for col in targets:
            if col not in df.columns:
                logger.warning(f"カラム {col} が存在しないためスキップします。")
                continue

            logger.info(f"カテゴリ {col} の集計中...")

            # 欠損値は 'unknown' として扱う
            # 元のデータを変更しないようにコピーするか、あるいはそのまま埋める
            # ここでは fillna しますが、元の列に影響を与えるので注意
            # ただし、IDが欠損していることは稀（スクレイピングの仕様上）
            if df[col].isnull().any():
                df[col] = df[col].fillna('unknown')

            grouped = df.groupby(col)

            # 1. 過去の出走回数 (Experience)
            # shift(1) することで今回のレースを含めない
            # expanding().count() は非NaNをカウントする
            # race_id は常に存在するのでカウントに使用
            history_count = grouped['race_id'].transform(lambda x: x.shift(1).expanding().count()).fillna(0)
            df[f'{col}_n_races'] = history_count

            # 2. 過去の勝利数 (Wins)
            wins = grouped['rank'].transform(lambda x: (x == 1).astype(int).shift(1).expanding().sum()).fillna(0)

            # 3. 過去の複勝数 (Top3)
            top3 = grouped['rank'].transform(lambda x: (x <= 3).astype(int).shift(1).expanding().sum()).fillna(0)

            # 4. 勝率・複勝率の計算
            # 0除算が発生した場合は 0 で埋める
            # (初出走の騎手などは 0/0 = NaN -> 0 となる)
            df[f'{col}_win_rate'] = (wins / history_count).fillna(0)
            df[f'{col}_top3_rate'] = (top3 / history_count).fillna(0)

        logger.info("カテゴリ集計特徴量の生成完了")
        return df

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

        # ターゲットのカラム
        targets = ['jockey_id', 'trainer_id', 'sire_id']

        # マージ用のキーを退避
        if 'race_id' not in df.columns or 'date' not in df.columns:
             logger.error("race_id または date カラムがありません。")
             return df

        for col in targets:
            if col not in df.columns:
                logger.warning(f"カラム {col} が存在しないためスキップします。")
                continue

            logger.info(f"カテゴリ {col} の集計中...")

            # 欠損値対応
            if df[col].isnull().any():
                df[col] = df[col].fillna('unknown')

            # ---------------------------------------------------------
            # リーク防止ロジック:
            # 1. レース単位(race_id)・カテゴリ単位(col)での成績を集計する
            #    (同じレースに出走している同カテゴリの馬たちをまとめる)
            # ---------------------------------------------------------

            # 必要なカラムだけ抽出
            # rank列が必要 (1着=1, 3着以内<=3)
            tmp = df[[col, 'race_id', 'date', 'rank']].copy()

            # 各レース・カテゴリごとの成績
            # count: 出走数
            # wins: 1着数
            # top3: 3着内数
            tmp['is_win'] = (tmp['rank'] == 1).astype(int)
            tmp['is_top3'] = (tmp['rank'] <= 3).astype(int)

            # Group by race_id AND category
            race_stats = tmp.groupby(['race_id', col]).agg({
                'date': 'min',
                'rank': 'count', # 出走数
                'is_win': 'sum',
                'is_top3': 'sum'
            }).rename(columns={'rank': 'count'}).reset_index()

            # ---------------------------------------------------------
            # 2. 時系列順に並べて累積和をとる (Lag特徴量)
            # ---------------------------------------------------------

            # 日付順、レースID順にソート
            race_stats = race_stats.sort_values(['date', 'race_id'])

            grouped_stats = race_stats.groupby(col)

            # shift(1) して expanding sum
            # これにより「今のレース」を含まない、過去の累積が得られる
            race_stats[f'{col}_n_races'] = grouped_stats['count'].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
            race_stats[f'{col}_n_wins'] = grouped_stats['is_win'].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)
            race_stats[f'{col}_n_top3'] = grouped_stats['is_top3'].transform(lambda x: x.shift(1).expanding().sum()).fillna(0)

            # ---------------------------------------------------------
            # 3. 率の計算とマージ
            # ---------------------------------------------------------

            # 勝率・複勝率
            race_stats[f'{col}_win_rate'] = (race_stats[f'{col}_n_wins'] / race_stats[f'{col}_n_races']).fillna(0)
            race_stats[f'{col}_top3_rate'] = (race_stats[f'{col}_n_top3'] / race_stats[f'{col}_n_races']).fillna(0)

            # マージ用のカラムだけ残す
            merge_cols = ['race_id', col, f'{col}_n_races', f'{col}_win_rate', f'{col}_top3_rate']
            stats_to_merge = race_stats[merge_cols]

            # 元データにマージ
            # how='left' で、元の行数を変えない
            df = pd.merge(df, stats_to_merge, on=['race_id', col], how='left')

            # マージ後にNaNが出る場合（そのカテゴリが過去になかった場合など）は0埋め
            df[f'{col}_n_races'] = df[f'{col}_n_races'].fillna(0)
            df[f'{col}_win_rate'] = df[f'{col}_win_rate'].fillna(0)
            df[f'{col}_top3_rate'] = df[f'{col}_top3_rate'].fillna(0)

        logger.info("カテゴリ集計特徴量の生成完了")
        return df

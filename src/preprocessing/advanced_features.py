import pandas as pd
import logging

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """
    レース展開や血統など、より高度なドメイン知識に基づく特徴量を生成するクラス。
    """
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        高度な特徴量を追加します。

        Args:
            df (pd.DataFrame): 前処理済みのデータ。

        Returns:
            pd.DataFrame: 特徴量が追加されたデータ。
        """
        logger.info("高度特徴量の生成を開始...")

        # 1. 逃げ判定 (Is Nige)
        # passing_rank (通過順) は "1-1-1" や "10-10-9" のような文字列
        # 最初の数字が 1 なら「逃げた」と判定
        def is_nige(s):
            if not isinstance(s, str): return 0
            try:
                # 数字以外の文字が含まれる場合もあるので注意
                first_pos = s.split('-')[0]
                # "1(2)" のようなケースもあるかも？数値変換できるか試す
                if first_pos.isdigit():
                    return 1 if int(first_pos) == 1 else 0
                return 0
            except:
                return 0

        # passing_rankカラムが存在し、有効なデータがあるかチェック
        # 推論時はpassing_rankがNULLなので、その場合は0で初期化
        if 'passing_rank' in df.columns and df['passing_rank'].notna().any():
            # 一時的なカラムとして作成
            df['is_nige_temp'] = df['passing_rank'].apply(is_nige)
        else:
            # passing_rankがない場合（推論時など）は0で初期化
            logger.info("passing_rankが利用できないため、is_nige_tempを0で初期化します")
            df['is_nige_temp'] = 0


        # 2. 馬ごとの過去の逃げ率 (Nige Rate)
        # 時系列順にソートして集計
        df = df.sort_values(['horse_id', 'date'])
        grouped_horse = df.groupby('horse_id')

        # shift(1)してexpanding meanを計算 (リーク防止)
        df['nige_rate'] = grouped_horse['is_nige_temp'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)

        # 3. レース展開予測 (Race Pace)
        # そのレースに出走するメンバーの「逃げ率」を集計
        # これにより「逃げ馬が多い＝ハイペース？」などの予測が可能になる
        # nige_rate は過去データ由来なので、現在のレース内の集計に使ってもOK

        grouped_race = df.groupby('race_id')

        # メンバーの平均逃げ率
        df['race_avg_nige_rate'] = grouped_race['nige_rate'].transform('mean')

        # 逃げ馬候補の数 (逃げ率 0.5 以上の馬の数)
        df['race_nige_horse_count'] = grouped_race['nige_rate'].transform(lambda x: (x >= 0.5).sum())

        # 不要な一時カラムを削除
        df.drop(columns=['is_nige_temp'], inplace=True)

        logger.info("高度特徴量の生成完了")
        return df

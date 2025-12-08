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

        # 時系列順であることを保証
        df = df.sort_values(['date', 'race_id'])

        # ----------------------------------------------------------------
        # 0. 準備: 条件カラムの作成
        # ----------------------------------------------------------------
        # 距離区分 (Distance Category)
        # Sprint: <1400, Mile: <1900, Intermediate: <2400, Long: >=2400
        # distanceカラムが数値であることを前提
        if 'distance' in df.columns:
            df['distance_cat'] = pd.cut(
                df['distance'], 
                bins=[0, 1399, 1899, 2399, 9999], 
                labels=['Sprint', 'Mile', 'Intermediate', 'Long']
            )
        else:
            logger.warning("distanceカラムがないため、距離別集計をスキップします。")
            df['distance_cat'] = 'Unknown'

        # ----------------------------------------------------------------
        # 1. 基本集計 (Overall Stats)
        # ----------------------------------------------------------------
        targets = ['jockey_id', 'trainer_id', 'sire_id']
        for col in targets:
            df = self._aggregate_basic(df, col)

        # ----------------------------------------------------------------
        # 2. 条件別集計 (Context-Specific Stats)
        # ----------------------------------------------------------------
        # (1) 騎手 × コース (Jockey x Course)
        # venue (keibajo_code) を使用
        if 'venue' in df.columns:
            df = self._aggregate_context(df, 'jockey_id', 'venue', 'course')
            df = self._aggregate_context(df, 'sire_id', 'venue', 'course')
            df = self._aggregate_context(df, 'trainer_id', 'venue', 'course')

        # (2) 種牡馬 × 距離区分 (Sire x Distance)
        if 'sire_id' in df.columns:
            df = self._aggregate_context(df, 'sire_id', 'distance_cat', 'dist')

        if 'surface' in df.columns and 'sire_id' in df.columns:
             df = self._aggregate_context(df, 'sire_id', 'surface', 'track')

        # NEW: (4) 騎手 × conditions
        if 'jockey_id' in df.columns:
            if 'surface' in df.columns:
                df = self._aggregate_context(df, 'jockey_id', 'surface', 'surface')
            if 'distance_cat' in df.columns:
                df = self._aggregate_context(df, 'jockey_id', 'distance_cat', 'dist')

        # NEW: (5) 調教師 × conditions
        if 'trainer_id' in df.columns:
            if 'surface' in df.columns:
                df = self._aggregate_context(df, 'trainer_id', 'surface', 'surface')
            if 'distance_cat' in df.columns:
                df = self._aggregate_context(df, 'trainer_id', 'distance_cat', 'dist')

        # (4) 騎手 × 調教師 (Jockey x Trainer - Interaction)
        if 'trainer_id' in df.columns and 'jockey_id' in df.columns:
            df = self._aggregate_context(df, 'jockey_id', 'trainer_id', 'trainer')

        logger.info("カテゴリ集計特徴量の生成完了")
        # 一時カラム削除
        if 'distance_cat' in df.columns:
            df.drop(columns=['distance_cat'], inplace=True)
            
        return df

    def _aggregate_basic(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        if col not in df.columns:
            return df

        if df[col].isnull().any():
            df[col] = df[col].fillna('unknown')

        grouped = df.groupby(col)

        # shift(1)してexpanding集計 (リーク防止)
        # race_idカウントで出走回数
        # rank=1で勝利数、rank<=3で複勝数
        
        # transform内でshift(1) -> expandingを使う
        history_count = grouped['race_id'].transform(lambda x: x.shift(1).expanding().count()).fillna(0)
        wins = grouped['rank'].transform(lambda x: (x == 1).astype(int).shift(1).expanding().sum()).fillna(0)
        top3 = grouped['rank'].transform(lambda x: (x <= 3).astype(int).shift(1).expanding().sum()).fillna(0) # bugfix: rank <= 3

        df[f'{col}_n_races'] = history_count
        # 0除算回避
        df[f'{col}_win_rate'] = (wins / (history_count + 1e-5))
        df[f'{col}_top3_rate'] = (top3 / (history_count + 1e-5))
        
        return df

    def _aggregate_context(self, df: pd.DataFrame, target_col: str, cond_col: str, suffix: str) -> pd.DataFrame:
        """
        target_col x cond_col の組み合わせで集計
        例: jockey_id x keibajo_code -> jockey_course_win_rate
        """
        if target_col not in df.columns or cond_col not in df.columns:
            return df
            
        # 欠損埋め
        if df[target_col].isnull().any(): 
            df[target_col] = df[target_col].fillna('unknown')
        if df[cond_col].isnull().any():
            df[cond_col] = df[cond_col].fillna('unknown')

        grouped = df.groupby([target_col, cond_col])
        
        feature_prefix = f"{target_col.replace('_id', '')}_{suffix}" # jockey_course

        history_count = grouped['race_id'].transform(lambda x: x.shift(1).expanding().count()).fillna(0)
        wins = grouped['rank'].transform(lambda x: (x == 1).astype(int).shift(1).expanding().sum()).fillna(0)
        top3 = grouped['rank'].transform(lambda x: (x <= 3).astype(int).shift(1).expanding().sum()).fillna(0)

        # 出走回数が少ないとノイズになるので、信頼度のようなものを考慮したいが、
        # ここでは生の率を出す。モデルが回数(n_races)も見て判断することを期待。
        df[f'{feature_prefix}_n_races'] = history_count
        df[f'{feature_prefix}_win_rate'] = (wins / (history_count + 1e-5))
        df[f'{feature_prefix}_top3_rate'] = (top3 / (history_count + 1e-5))

        return df

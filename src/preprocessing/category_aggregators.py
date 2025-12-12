import pandas as pd
import logging

logger = logging.getLogger(__name__)

class CategoryAggregator:
    """
    カテゴリ変数（騎手、調教師、種牡馬など）の過去成績を集計するクラス。
    ターゲットエンコーディングに近いですが、リークを防ぐために過去データのみを使用します。
    
    [2025-12-12 リファクタリング]
    同一レース内のリーク問題を修正。race_id単位で事前集約することで、
    同じレースに出走する同カテゴリの馬の結果が特徴量に含まれることを防止。
    詳細: docs/refactoring_log/2025-12-12_data_leakage_fix/CHANGELOG.md
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
        df = df.sort_values(['date', 'race_id']).reset_index(drop=True)
        
        # 元のインデックスを保持(マージ用)
        df['_original_idx'] = df.index

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
        targets = ['jockey_id', 'trainer_id', 'sire_id', 'class_level']
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
            
            # v7新規: 騎手×調教師の騎乗回数 (trainer_jockey_count)
            # 既存のjockey_trainer_n_racesと同じだが、より直感的な名前で追加
            df['trainer_jockey_count'] = df['jockey_trainer_n_races'].copy() if 'jockey_trainer_n_races' in df.columns else 0

        # ----------------------------------------------------------------
        # v7 新規: 母父 (BMS) × 距離区分
        # ----------------------------------------------------------------
        if 'bms_id' in df.columns and 'distance_cat' in df.columns:
            logger.info("v7: 母父(BMS) × 距離 の集計...")
            df = self._aggregate_context(df, 'bms_id', 'distance_cat', 'dist')

        logger.info("カテゴリ集計特徴量の生成完了")
        # 一時カラム削除
        if 'distance_cat' in df.columns:
            df.drop(columns=['distance_cat'], inplace=True)
        if '_original_idx' in df.columns:
            df.drop(columns=['_original_idx'], inplace=True)
            
        return df

    def _aggregate_basic(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        基本的なカテゴリ集計（リークなし版）
        
        [修正ポイント]
        1. まず race_id × カテゴリ で集約し、1レースにつき1レコードにする
        2. その上で shift(1) + expanding() で累積統計を計算
        3. 元のDataFrameにマージして戻す
        
        これにより、同一レース内の複数馬（同じ騎手等）が互いの結果を参照することを防ぐ
        """
        if col not in df.columns:
            return df

        if df[col].isnull().any():
            df[col] = df[col].fillna('unknown')

        # ----------------------------------------------------------------
        # Step 1: レース単位で集約（同一レース内の重複を排除）
        # ----------------------------------------------------------------
        # 同じレースに同じ騎手が複数騎乗している場合、1レコードにまとめる
        race_agg = df.groupby(['race_id', col, 'date'], observed=True).agg({
            'rank': [
                ('wins', lambda x: (x == 1).sum()),      # 勝利数
                ('top3', lambda x: (x <= 3).sum()),      # 複勝数
                ('count', 'count')                       # 出走頭数
            ]
        }).reset_index()
        race_agg.columns = ['race_id', col, 'date', 'wins', 'top3', 'count']
        race_agg = race_agg.sort_values('date')
        
        # ----------------------------------------------------------------
        # Step 2: カテゴリごとの累積統計（shift(1)でリーク防止）
        # ----------------------------------------------------------------
        grouped = race_agg.groupby(col, observed=True)
        race_agg['cum_races'] = grouped['count'].transform(
            lambda x: x.shift(1).expanding().sum()
        ).fillna(0)
        race_agg['cum_wins'] = grouped['wins'].transform(
            lambda x: x.shift(1).expanding().sum()
        ).fillna(0)
        race_agg['cum_top3'] = grouped['top3'].transform(
            lambda x: x.shift(1).expanding().sum()
        ).fillna(0)
        
        # ----------------------------------------------------------------
        # Step 3: 元のDataFrameにマージ
        # ----------------------------------------------------------------
        merge_cols = ['race_id', col]
        stats_df = race_agg[['race_id', col, 'cum_races', 'cum_wins', 'cum_top3']].copy()
        
        # マージ前に既存の列があれば削除
        for c in [f'{col}_n_races', f'{col}_win_rate', f'{col}_top3_rate']:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
        
        df = df.merge(stats_df, on=merge_cols, how='left')
        
        df[f'{col}_n_races'] = df['cum_races'].fillna(0)
        df[f'{col}_win_rate'] = df['cum_wins'].fillna(0) / (df['cum_races'].fillna(0) + 1e-5)
        df[f'{col}_top3_rate'] = df['cum_top3'].fillna(0) / (df['cum_races'].fillna(0) + 1e-5)
        
        # 一時列を削除
        df.drop(columns=['cum_races', 'cum_wins', 'cum_top3'], inplace=True)
        
        return df

    def _aggregate_context(self, df: pd.DataFrame, target_col: str, cond_col: str, suffix: str) -> pd.DataFrame:
        """
        target_col x cond_col の組み合わせで集計（リークなし版）
        例: jockey_id x keibajo_code -> jockey_course_win_rate
        
        [2025-12-12 修正]
        _aggregate_basicと同様、race_id単位で事前集約してからマージする方式に変更。
        """
        if target_col not in df.columns or cond_col not in df.columns:
            return df
            
        # 欠損埋め
        if df[target_col].isnull().any(): 
            df[target_col] = df[target_col].fillna('unknown')
        if df[cond_col].isnull().any():
            df[cond_col] = df[cond_col].fillna('unknown')

        feature_prefix = f"{target_col.replace('_id', '')}_{suffix}"  # jockey_course

        # ----------------------------------------------------------------
        # Step 1: レース単位で集約
        # ----------------------------------------------------------------
        race_agg = df.groupby(['race_id', target_col, cond_col, 'date'], observed=True).agg({
            'rank': [
                ('wins', lambda x: (x == 1).sum()),
                ('top3', lambda x: (x <= 3).sum()),
                ('count', 'count')
            ]
        }).reset_index()
        race_agg.columns = ['race_id', target_col, cond_col, 'date', 'wins', 'top3', 'count']
        race_agg = race_agg.sort_values('date')
        
        # ----------------------------------------------------------------
        # Step 2: カテゴリ×条件ごとの累積統計
        # ----------------------------------------------------------------
        grouped = race_agg.groupby([target_col, cond_col], observed=True)
        race_agg['cum_races'] = grouped['count'].transform(
            lambda x: x.shift(1).expanding().sum()
        ).fillna(0)
        race_agg['cum_wins'] = grouped['wins'].transform(
            lambda x: x.shift(1).expanding().sum()
        ).fillna(0)
        race_agg['cum_top3'] = grouped['top3'].transform(
            lambda x: x.shift(1).expanding().sum()
        ).fillna(0)
        
        # ----------------------------------------------------------------
        # Step 3: 元のDataFrameにマージ
        # ----------------------------------------------------------------
        merge_cols = ['race_id', target_col, cond_col]
        stats_df = race_agg[['race_id', target_col, cond_col, 'cum_races', 'cum_wins', 'cum_top3']].copy()
        
        # マージ前に既存の列があれば削除
        for c in [f'{feature_prefix}_n_races', f'{feature_prefix}_win_rate', f'{feature_prefix}_top3_rate']:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
        
        df = df.merge(stats_df, on=merge_cols, how='left')
        
        # 出走回数が少ないとノイズになるので、信頼度のようなものを考慮したいが、
        # ここでは生の率を出す。モデルが回数(n_races)も見て判断することを期待。
        df[f'{feature_prefix}_n_races'] = df['cum_races'].fillna(0)
        df[f'{feature_prefix}_win_rate'] = df['cum_wins'].fillna(0) / (df['cum_races'].fillna(0) + 1e-5)
        df[f'{feature_prefix}_top3_rate'] = df['cum_top3'].fillna(0) / (df['cum_races'].fillna(0) + 1e-5)
        
        # 一時列を削除
        df.drop(columns=['cum_races', 'cum_wins', 'cum_top3'], inplace=True)

        return df

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RelativeFeatureEngineer:
    """
    レース内での相対的な特徴量を生成するクラス。
    
    絶対値ではなく、レース内での相対的な位置（偏差値、順位）を特徴量化することで、
    レース難易度の違いを吸収し、モデルの汎化性能を向上させる。
    """
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """相対的特徴量を追加"""
        logger.info("相対的特徴量の生成を開始...")
        
        # 新しい特徴量を格納する辞書
        new_cols = {}
        
        # =============================================================
        # 1. レース内標準化（偏差値化）
        # =============================================================
        self._add_deviation_scores(df, new_cols)
        
        # =============================================================
        # 2. レース内順位
        # =============================================================
        self._add_race_ranks(df, new_cols)
        
        # =============================================================
        # 3. 相対値（レース平均との差）
        # =============================================================
        self._add_relative_values(df, new_cols)
        
        # まとめて結合
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            df = pd.concat([df, new_df], axis=1)
        
        logger.info("相対的特徴量の生成完了")
        return df
    
    def _add_deviation_scores(self, df: pd.DataFrame, new_cols: dict) -> None:
        """
        レース内偏差値を計算
        
        偏差値 = 50 + 10 * (x - mean) / std
        """
        # 偏差値化する対象カラム
        deviation_cols = []
        
        cols_to_check = [
            'horse_past_avg_rank', 'jockey_id_win_rate', 'weight', 'age',
            'horse_past_avg_time', 'trainer_id_win_rate', 'impost'
        ]
        
        for col in cols_to_check:
            if col in df.columns:
                deviation_cols.append(col)
        
        # 偏差値計算
        if deviation_cols:
            # Ensure numeric to prevent TypeError on strings
            # We create a temp view/copy to not affect original df if we don't want to?
            # Actually, fixing types in df is good.
            for col in deviation_cols:
                # errors='coerce' turns non-numeric to NaN
                # We check compatibility first? No, just force.
                # Only if object type?
                if df[col].dtype == 'object':
                     df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # GroupBy object
            g = df.groupby('race_id')[deviation_cols]
            
            means = g.transform('mean')
            stds = g.transform('std')
            
            for col in deviation_cols:
                z = (df[col] - means[col]) / (stds[col] + 1e-8)
                new_cols[f'{col}_deviation'] = 50 + 10 * z

    def _add_race_ranks(self, df: pd.DataFrame, new_cols: dict) -> None:
        """
        レース内順位を計算
        """
        # (column, ascending)
        rank_targets = []
        
        if 'jockey_id_win_rate' in df.columns:
            rank_targets.append(('jockey_id_win_rate', False))
        if 'trainer_id_win_rate' in df.columns:
            rank_targets.append(('trainer_id_win_rate', False))
        if 'horse_past_avg_rank' in df.columns:
            rank_targets.append(('horse_past_avg_rank', True))
        
        for col, ascending in rank_targets:
            new_cols[f'{col}_race_rank'] = df.groupby('race_id')[col].rank(
                ascending=ascending,
                method='min',
                na_option='keep'
            )
    
    def _add_relative_values(self, df: pd.DataFrame, new_cols: dict) -> None:
        """
        レース平均との差分を計算
        """
        relative_cols = []
        cols_to_check = ['weight', 'age', 'horse_n_races', 'impost']
        
        for col in cols_to_check:
            if col in df.columns:
                relative_cols.append(col)
                
        if relative_cols:
            means = df.groupby('race_id')[relative_cols].transform('mean')
            
            for col in relative_cols:
                new_cols[f'{col}_relative'] = df[col] - means[col]

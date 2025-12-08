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
        
        # =============================================================
        # 1. レース内標準化（偏差値化）
        # =============================================================
        df = self._add_deviation_scores(df)
        
        # =============================================================
        # 2. レース内順位
        # =============================================================
        df = self._add_race_ranks(df)
        
        # =============================================================
        # 3. 相対値（レース平均との差）
        # =============================================================
        df = self._add_relative_values(df)
        
        logger.info("相対的特徴量の生成完了")
        return df
    
    def _add_deviation_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        レース内偏差値を計算
        
        偏差値 = 50 + 10 * (x - mean) / std
        
        これにより、グレードレースと未勝利戦での絶対値の違いを吸収できる。
        """
        # 偏差値化する対象カラム
        deviation_cols = []
        
        # 過去平均着順（存在する場合）
        if 'horse_past_avg_rank' in df.columns:
            deviation_cols.append('horse_past_avg_rank')
        
        # 騎手勝率
        if 'jockey_id_win_rate' in df.columns:
            deviation_cols.append('jockey_id_win_rate')
        
        # 馬体重
        if 'weight' in df.columns:
            deviation_cols.append('weight')
        
        # 年齢
        if 'age' in df.columns:
            deviation_cols.append('age')
        
        # 過去平均タイム（存在する場合）
        if 'horse_past_avg_time' in df.columns:
            deviation_cols.append('horse_past_avg_time')
        
        # 調教師勝率
        if 'trainer_id_win_rate' in df.columns:
            deviation_cols.append('trainer_id_win_rate')
        
        # 偏差値計算
        for col in deviation_cols:
            df[f'{col}_deviation'] = df.groupby('race_id')[col].transform(
                lambda x: 50 + 10 * (x - x.mean()) / (x.std() + 1e-8)
            )
        
        return df
    
    def _add_race_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        レース内順位を計算
        
        各特徴量について、そのレース内での順位を付ける。
        （降順: 値が大きいほど順位が高い）
        NOTE: オッズ・人気はリークになるため除外 (Phase 11.1 fix)
        """
        # 順位化する対象カラム
        rank_cols = []
        
        # 騎手勝率
        if 'jockey_id_win_rate' in df.columns:
            rank_cols.append(('jockey_id_win_rate', False))  # 降順（高いほど良い）
        
        # 調教師勝率
        if 'trainer_id_win_rate' in df.columns:
            rank_cols.append(('trainer_id_win_rate', False))
        
        # 過去平均着順（低いほど良い）
        if 'horse_past_avg_rank' in df.columns:
            rank_cols.append(('horse_past_avg_rank', True))  # 昇順
        
        # 順位計算
        for col, ascending in rank_cols:
            df[f'{col}_race_rank'] = df.groupby('race_id')[col].rank(
                ascending=ascending,
                method='min',  # 同率の場合は最小順位
                na_option='keep'  # NaNはNaNのまま
            )
        
        return df
    
    def _add_relative_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        レース平均との差分を計算
        
        馬体重など、絶対値よりも相対値が重要な特徴量について計算。
        """
        # 相対値化する対象カラム
        relative_cols = []
        
        # 馬体重
        if 'weight' in df.columns:
            relative_cols.append('weight')
        
        # 年齢
        if 'age' in df.columns:
            relative_cols.append('age')
        
        # 過去走数
        if 'horse_n_races' in df.columns:
            relative_cols.append('horse_n_races')
        
        # 相対値計算（レース平均との差）
        for col in relative_cols:
            race_mean = df.groupby('race_id')[col].transform('mean')
            df[f'{col}_relative'] = df[col] - race_mean
        
        return df

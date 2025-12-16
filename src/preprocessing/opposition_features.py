"""
opposition_features.py - 対戦レベル特徴量生成モジュール

[v11 Extended N3]
レースの相手関係の強さを数値化する。

opponent_strength = (レース内strength合計 - 自馬strength) / (頭数 - 1)

リーク防止:
- 使用する strength 列は必ず shift済み（当該レースを含まない）であること
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class OppositionFeatureEngineer:
    """
    対戦レベル: レースの相手関係の強さを数値化
    
    特徴量:
    - race_opponent_strength: 同一レースの他馬の平均事前strength（自分除外）
    - relative_strength: 自馬strengthと相手平均の差
    """
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        対戦レベル特徴量を追加します。
        
        Args:
            df: mean_rank_all（shift済み通算平均着順）を含むDataFrame
        
        Returns:
            対戦レベル特徴量が追加されたDataFrame
        """
        logger.info("v11 N3: 対戦レベル特徴量を生成中...")
        
        # 使用する strength 列（shift済みであることを前提）
        strength_col = 'mean_rank_all'  # 通算平均着順（aggregators.py で shift(1).expanding()）
        
        if strength_col not in df.columns:
            logger.warning(f"{strength_col} カラムがないため、対戦レベル特徴量をスキップします")
            return df
        
        # strength が NaN の馬は除外して計算（新馬等）
        df[strength_col] = pd.to_numeric(df[strength_col], errors='coerce')
        
        # レース内の合計と頭数を計算
        race_sum = df.groupby('race_id')[strength_col].transform('sum')
        race_count = df.groupby('race_id')[strength_col].transform('count')
        
        # 自分を除外した相手の平均
        # opponent_strength = (sum - self) / (n - 1)
        df['race_opponent_strength'] = (
            (race_sum - df[strength_col]) / (race_count - 1).clip(lower=1)
        )
        
        # n=1（単頭レース）や新馬（strength=NaN）の場合はNaN→ニュートラル＋missing flag
        df['race_opponent_strength_is_missing'] = (
            (race_count <= 1) | df['race_opponent_strength'].isna()
        ).astype('int8')
        
        # ニュートラル値: 全体の中央値
        neutral_val = df[strength_col].median()
        if pd.isna(neutral_val):
            neutral_val = 8.0  # フォールバック
        
        df['race_opponent_strength'] = df['race_opponent_strength'].fillna(neutral_val).astype('float32')
        
        # 自馬との相対差（小さいほど相手が弱い = 有利）
        df['relative_strength'] = (df[strength_col] - df['race_opponent_strength']).astype('float32')
        
        logger.info(f"対戦レベル特徴量生成完了: race_opponent_strength, relative_strength")
        logger.info(f"  strength中央値: {neutral_val:.2f}, 欠損: {df['race_opponent_strength_is_missing'].sum()}件")
        
        return df

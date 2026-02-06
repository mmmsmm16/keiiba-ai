"""
前走レースレベル評価用の特徴量生成クラス

Feature Engineering v7で追加。
「前走で強い相手と戦っていたか」を数値化する。

生成する特徴量:
- lag1_race_avg_rank: 前走のメンバー平均着順
- lag1_winner_class: 前走の勝ち馬のクラスレベル
- race_member_strength: レース出走メンバーの総合力スコア
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RaceLevelFeatureEngineer:
    """
    前走のレースレベルを評価する特徴量を生成するクラス。
    「前走で強い相手と戦っていたか」を数値化する。
    
    注意: この計算は時系列的に複雑で、厳密には未来情報を参照する可能性があるため、
    シンプルな代替指標を使用する。
    """
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("レースレベル特徴量の生成を開始...")
        
        # ソート（時系列順）
        df = df.sort_values(['horse_id', 'date']).copy()
        
        # ------------------------------------------------------
        # 1. 各レースのメンバー強度スコア (Race Member Strength)
        # ------------------------------------------------------
        # レース出走メンバーの平均「過去成績」をスコア化
        # mean_rank_all が存在する場合、それを使用
        logger.info("1. レースメンバー強度スコアの計算...")
        
        if 'mean_rank_all' in df.columns:
            # レース内のmean_rank_allの平均（メンバーの平均実力）
            df['race_member_strength'] = df.groupby('race_id')['mean_rank_all'].transform('mean')
            df['race_member_strength'] = df['race_member_strength'].fillna(8.0).astype('float32')
            
            # レース内での自分の相対実力（メンバー平均との差）
            df['relative_strength'] = (df['mean_rank_all'] - df['race_member_strength']).fillna(0).astype('float32')
        
        # ------------------------------------------------------
        # 2. 前走のレースメンバー強度 (Lag1 Race Member Strength)
        # ------------------------------------------------------
        logger.info("2. 前走レースメンバー強度の計算...")
        
        if 'race_member_strength' in df.columns:
            # 馬ごとの前走のrace_member_strengthを取得
            df['lag1_race_member_strength'] = df.groupby('horse_id')['race_member_strength'].shift(1)
            df['lag1_race_member_strength'] = df['lag1_race_member_strength'].fillna(8.0).astype('float32')
        
        # ------------------------------------------------------
        # 3. 前走の着順とレースレベルの組み合わせスコア
        # ------------------------------------------------------
        # 強いメンバーのレースで好走 = 価値が高い
        logger.info("3. 前走価値スコアの計算...")
        
        if 'lag1_rank' in df.columns and 'lag1_race_member_strength' in df.columns:
            # スコア = (メンバー強度が高いほど良い) × (着順が良いほど良い)
            # メンバー強度: 低い = 強い (mean_rankなので)
            # 着順: 低い = 良い
            # 
            # 価値スコア = 1 / (lag1_rank + 1) * (1 / lag1_race_member_strength)
            # → 強いメンバーで好着順 = 高スコア
            
            lag1_rank = df['lag1_rank'].fillna(10)
            lag1_strength = df['lag1_race_member_strength'].fillna(8)
            
            # シンプルな計算式
            # 強いメンバー(低い値) × 好着順(低い値) = 良いパフォーマンス
            df['lag1_performance_value'] = (
                (10 - lag1_rank.clip(1, 10)) / 10 *  # 着順スコア (高いほど良い)
                (10 - lag1_strength.clip(1, 10)) / 10  # メンバー強度スコア (強いほど良い)
            ).astype('float32')
        
        # ------------------------------------------------------
        # 4. 前走の出走頭数 (難易度指標)
        # ------------------------------------------------------
        logger.info("4. 前走頭数の計算...")
        
        if 'n_horses' in df.columns:
            df['lag1_n_horses'] = df.groupby('horse_id')['n_horses'].shift(1)
            df['lag1_n_horses'] = df['lag1_n_horses'].fillna(14).astype('int8')
        
        logger.info("レースレベル特徴量の生成完了")
        return df

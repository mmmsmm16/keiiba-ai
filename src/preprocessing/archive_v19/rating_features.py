"""
rating_features.py - 馬のEloレーティング特徴量生成モジュール

[v12 P2] 馬の総合力を1つのスコアに圧縮する。
オッズを使わないモデルで重要な「相手関係込みの能力評価」を提供。

[特徴量]
- horse_rating: レース前時点のEloレーティング
- rating_uncertainty: サンプル数が少ないほど高い（不確実性）
- rating_change_last: 前走でのレーティング変化量
- field_avg_rating: レース内全馬の平均レーティング
- field_max_rating: レース内最高レーティング
- horse_vs_field: horse_rating - field_avg_rating

[計算ロジック]
- 初期レーティング: 1500
- K-factor: 32
- 期待勝率: 1 / (1 + 10^((opponent_rating - horse_rating) / 400))
- 更新: new_rating = old_rating + K * (actual - expected)
  - actual = 1.0 if won, else based on normalized position

[リーク防止]
- shift(1) で当該レース結果を除外
- レース単位で事前にレーティング計算後、そのレースで更新
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Elo パラメータ
INITIAL_RATING = 1500.0
K_FACTOR = 32.0
RATING_DIVISOR = 400.0  # 標準的なElo係数


class RatingFeatureEngineer:
    """
    馬のEloレーティングを計算し、特徴量として付与するクラス。
    """
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Eloレーティング特徴量を追加します。
        
        Args:
            df: 日付順にソートされたDataFrame（horse_id, race_id, rank が必要）
            
        Returns:
            レーティング特徴量が追加されたDataFrame
        """
        logger.info("Eloレーティング特徴量の生成を開始...")
        
        # 必須カラムチェック
        required = ['horse_id', 'race_id', 'date', 'rank']
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"Eloレーティング: 必要なカラムがありません: {missing}")
            return df
        
        # 日付順ソート（重要！）
        df = df.sort_values(['date', 'race_id']).reset_index(drop=True)
        
        # レーティング辞書: horse_id -> (current_rating, n_races)
        ratings = {}
        
        # 結果格納用
        rating_before = []
        rating_after = []
        n_races_list = []
        
        # レースごとにグループ化して処理
        race_groups = df.groupby('race_id', sort=False)
        
        for race_id, race_df in race_groups:
            # このレースの馬のレーティングを取得（レース前）
            horse_ids = race_df['horse_id'].tolist()
            ranks = race_df['rank'].tolist()
            
            # 各馬の現在レーティングを取得
            before_ratings = []
            before_n_races = []
            for hid in horse_ids:
                if hid in ratings:
                    r, n = ratings[hid]
                else:
                    r, n = INITIAL_RATING, 0
                before_ratings.append(r)
                before_n_races.append(n)
            
            rating_before.extend(before_ratings)
            n_races_list.extend(before_n_races)
            
            # フィールド平均レーティング
            field_avg = np.mean(before_ratings)
            
            # レース結果に基づいてレーティング更新
            n_horses = len(horse_ids)
            after_ratings = []
            
            for i, (hid, rank, old_r) in enumerate(zip(horse_ids, ranks, before_ratings)):
                # 期待スコア: 全対戦相手との合計
                expected = 0.0
                for j, opp_r in enumerate(before_ratings):
                    if i != j:
                        # 1対1の期待勝率
                        exp_win = 1.0 / (1.0 + 10 ** ((opp_r - old_r) / RATING_DIVISOR))
                        expected += exp_win
                
                # 期待スコアを正規化 (0-1)
                expected = expected / max(n_horses - 1, 1)
                
                # 実際のスコア: 順位を0-1に変換 (1着=1.0, 最下位=0.0)
                # rank=0やNaNは除外対象だが、安全のため処理
                if pd.isna(rank) or rank <= 0 or rank > n_horses:
                    actual = 0.5  # ニュートラル
                else:
                    actual = 1.0 - (rank - 1) / max(n_horses - 1, 1)
                
                # レーティング更新
                new_r = old_r + K_FACTOR * (actual - expected)
                after_ratings.append(new_r)
                
                # 辞書更新
                old_n = before_n_races[i]
                ratings[hid] = (new_r, old_n + 1)
            
            rating_after.extend(after_ratings)
        
        # DataFrameに追加
        df['horse_rating'] = rating_before
        df['_rating_after'] = rating_after
        df['rating_n_races'] = n_races_list
        
        # rating_uncertainty: サンプル数が少ないほど高い (1 / sqrt(n + 1))
        df['rating_uncertainty'] = 1.0 / np.sqrt(df['rating_n_races'] + 1)
        
        # rating_change_last: 前走でのレーティング変化
        # 現在の _rating_after から次走の horse_rating を引く...これは複雑
        # 代わりに: shift(1) した rating_after - shift(1) した rating_before
        df['_prev_rating_after'] = df.groupby('horse_id')['_rating_after'].shift(1)
        df['_prev_rating_before'] = df.groupby('horse_id')['horse_rating'].shift(1)
        df['rating_change_last'] = df['_prev_rating_after'] - df['_prev_rating_before']
        df['rating_change_last'] = df['rating_change_last'].fillna(0).astype('float32')
        
        # Field Strength 特徴量
        logger.info("フィールド強度特徴量を生成中...")
        df['field_avg_rating'] = df.groupby('race_id')['horse_rating'].transform('mean')
        df['field_max_rating'] = df.groupby('race_id')['horse_rating'].transform('max')
        df['horse_vs_field'] = df['horse_rating'] - df['field_avg_rating']
        
        # float32にダウンキャスト
        for col in ['horse_rating', 'rating_uncertainty', 'field_avg_rating', 'field_max_rating', 'horse_vs_field']:
            if col in df.columns:
                df[col] = df[col].astype('float32')
        
        # 一時カラム削除
        df.drop(columns=['_rating_after', '_prev_rating_after', '_prev_rating_before'], inplace=True, errors='ignore')
        
        # 統計ログ
        logger.info(f"Eloレーティング統計:")
        logger.info(f"  平均: {df['horse_rating'].mean():.1f}, 標準偏差: {df['horse_rating'].std():.1f}")
        logger.info(f"  最小: {df['horse_rating'].min():.1f}, 最大: {df['horse_rating'].max():.1f}")
        logger.info(f"  horse_vs_field 平均: {df['horse_vs_field'].mean():.2f}")
        
        logger.info("Eloレーティング特徴量の生成完了")
        return df

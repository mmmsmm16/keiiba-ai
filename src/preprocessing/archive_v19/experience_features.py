"""
コース経験・距離経験・初条件フラグの特徴量生成クラス

Feature Engineering v7で追加。
馬がこのコースや距離での経験があるか、初めての条件かを判定する。
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ExperienceFeatureEngineer:
    """
    コース経験・距離経験に関する特徴量を生成するクラス。
    
    生成する特徴量:
    - course_experience: このコース(venue + surface)での過去走数
    - course_best_rank: このコースでの最高着順
    - distance_experience: この距離帯での過去走数
    - distance_best_rank: この距離帯での最高着順
    - first_turf: 初芝フラグ
    - first_dirt: 初ダートフラグ
    - first_distance_cat: この距離カテゴリ初挑戦フラグ
    - jockey_change_flag: 騎手が前走から変わったかフラグ
    """
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("経験特徴量の生成を開始...")
        
        # ソート（時系列順）
        # Note: df自体をソートしてしまうと呼び出し元に影響する可能性があるが、
        # ここでは時系列計算が必要なのでソート済みdfを返す仕様とする
        df = df.sort_values(['horse_id', 'date']).copy()
        
        # 新しい特徴量を格納する辞書
        new_cols = {}

        # ------------------------------------------------------
        # 1. コース経験 (Course Experience)
        # ------------------------------------------------------
        logger.info("1. コース経験の計算...")
        
        if 'venue' in df.columns and 'surface' in df.columns:
            course_key = df['venue'].astype(str) + '_' + df['surface'].astype(str)
            
            # 過去走数
            new_cols['course_experience'] = df.groupby(['horse_id', course_key]).cumcount()
            
            # 過去最高着順
            if 'rank' in df.columns:
                new_cols['course_best_rank'] = df.groupby(['horse_id', course_key])['rank'].transform(
                    lambda x: x.shift(1).expanding().min()
                ).fillna(99).astype('float32')
            else:
                new_cols['course_best_rank'] = pd.Series(99.0, index=df.index, dtype='float32')
        else:
            logger.warning("venue/surfaceカラムがないため、コース経験はスキップします")
        
        # ------------------------------------------------------
        # 2. 距離経験 (Distance Experience)
        # ------------------------------------------------------
        logger.info("2. 距離経験の計算...")
        
        if 'distance' in df.columns:
            # 距離カテゴリ化
            dist_cat = pd.cut(
                df['distance'],
                bins=[0, 1400, 1800, 2200, 9999],
                labels=['sprint', 'mile', 'intermediate', 'long']
            )
            # 後でFeatureとして使うならdfに入れるべきだが、今回は計算用
            new_cols['distance_category'] = dist_cat
            
            # 過去走数
            new_cols['distance_experience'] = df.groupby(['horse_id', dist_cat], observed=True).cumcount()
            
            # 過去最高着順
            if 'rank' in df.columns:
                new_cols['distance_best_rank'] = df.groupby(['horse_id', dist_cat], observed=True)['rank'].transform(
                    lambda x: x.shift(1).expanding().min()
                ).fillna(99).astype('float32')
            else:
                new_cols['distance_best_rank'] = pd.Series(99.0, index=df.index, dtype='float32')
            
            # この距離カテゴリ初挑戦フラグ
            new_cols['first_distance_cat'] = (new_cols['distance_experience'] == 0).astype('int8')
        else:
            logger.warning("distanceカラムがないため、距離経験はスキップします")
        
        # ------------------------------------------------------
        # 3. 初条件フラグ (First Time Flags)
        # ------------------------------------------------------
        logger.info("3. 初条件フラグの計算...")
        
        if 'surface' in df.columns:
            # 芝の経験数
            # 芝の経験数
            # Use temporary series with same index
            is_turf = (df['surface'] == '芝')
            turf_exp = is_turf.groupby(df['horse_id']).cumsum() - is_turf.astype(int)
            new_cols['first_turf'] = (is_turf & (turf_exp == 0)).astype('int8')
            
            is_dirt = (df['surface'] == 'ダート')
            dirt_exp = is_dirt.groupby(df['horse_id']).cumsum() - is_dirt.astype(int)
            new_cols['first_dirt'] = (is_dirt & (dirt_exp == 0)).astype('int8')
        
        # ------------------------------------------------------
        # 4. 騎手乗り替わりフラグ (Jockey Change Flag)
        # ------------------------------------------------------
        logger.info("4. 騎手乗り替わりフラグの計算...")
        
        if 'jockey_id' in df.columns:
            prev_jockey_id = df.groupby('horse_id')['jockey_id'].shift(1)
            new_cols['jockey_change_flag'] = (
                (df['jockey_id'] != prev_jockey_id) & 
                (prev_jockey_id.notna())
            ).astype('int8')
        
        # ------------------------------------------------------
        # 5. 斤量経験（今回斤量 vs 過去最高斤量）
        # ------------------------------------------------------
        if 'impost' in df.columns:
            max_impost_before = df.groupby('horse_id')['impost'].transform(
                lambda x: x.shift(1).expanding().max()
            ).fillna(df['impost'])
            
            new_cols['is_career_high_impost'] = (df['impost'] > max_impost_before).astype('int8')
        
        # まとめて結合 (Fragmentation回避)
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            df = pd.concat([df, new_df], axis=1)
            
        logger.info("経験特徴量の生成完了")
        return df

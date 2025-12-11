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
        df = df.sort_values(['horse_id', 'date']).copy()
        
        # ------------------------------------------------------
        # 1. コース経験 (Course Experience)
        # ------------------------------------------------------
        logger.info("1. コース経験の計算...")
        
        # コースキーの生成 (venue + surface)
        if 'venue' in df.columns and 'surface' in df.columns:
            df['course_key'] = df['venue'].astype(str) + '_' + df['surface'].astype(str)
            
            # 過去走数 (自分自身は含めない = shift後のcumcount)
            df['course_experience'] = df.groupby(['horse_id', 'course_key']).cumcount()
            
            # 過去最高着順 (自分自身は含めない) - rankがない場合はスキップ
            if 'rank' in df.columns:
                df['course_best_rank'] = df.groupby(['horse_id', 'course_key'])['rank'].transform(
                    lambda x: x.shift(1).expanding().min()
                ).fillna(99).astype('float32')
            else:
                df['course_best_rank'] = 99.0
            
            # 一時キー削除
            del df['course_key']
        else:
            logger.warning("venue/surfaceカラムがないため、コース経験はスキップします")
        
        # ------------------------------------------------------
        # 2. 距離経験 (Distance Experience)
        # ------------------------------------------------------
        logger.info("2. 距離経験の計算...")
        
        if 'distance' in df.columns:
            # 距離カテゴリ化
            df['distance_category'] = pd.cut(
                df['distance'],
                bins=[0, 1400, 1800, 2200, 9999],
                labels=['sprint', 'mile', 'intermediate', 'long']
            )
            
            # 過去走数
            df['distance_experience'] = df.groupby(['horse_id', 'distance_category']).cumcount()
            
            # 過去最高着順 - rankがない場合はスキップ
            if 'rank' in df.columns:
                df['distance_best_rank'] = df.groupby(['horse_id', 'distance_category'])['rank'].transform(
                    lambda x: x.shift(1).expanding().min()
                ).fillna(99).astype('float32')
            else:
                df['distance_best_rank'] = 99.0
            
            # この距離カテゴリ初挑戦フラグ
            df['first_distance_cat'] = (df['distance_experience'] == 0).astype('int8')
        else:
            logger.warning("distanceカラムがないため、距離経験はスキップします")
        
        # ------------------------------------------------------
        # 3. 初条件フラグ (First Time Flags)
        # ------------------------------------------------------
        logger.info("3. 初条件フラグの計算...")
        
        if 'surface' in df.columns:
            # 芝の経験数
            df['_turf_count'] = (df['surface'] == '芝').astype(int)
            df['_turf_cumsum'] = df.groupby('horse_id')['_turf_count'].cumsum() - df['_turf_count']
            df['first_turf'] = ((df['surface'] == '芝') & (df['_turf_cumsum'] == 0)).astype('int8')
            
            # ダートの経験数
            df['_dirt_count'] = (df['surface'] == 'ダート').astype(int)
            df['_dirt_cumsum'] = df.groupby('horse_id')['_dirt_count'].cumsum() - df['_dirt_count']
            df['first_dirt'] = ((df['surface'] == 'ダート') & (df['_dirt_cumsum'] == 0)).astype('int8')
            
            # 一時カラム削除
            for c in ['_turf_count', '_turf_cumsum', '_dirt_count', '_dirt_cumsum']:
                if c in df.columns:
                    del df[c]
        
        # ------------------------------------------------------
        # 4. 騎手乗り替わりフラグ (Jockey Change Flag)
        # ------------------------------------------------------
        logger.info("4. 騎手乗り替わりフラグの計算...")
        
        if 'jockey_id' in df.columns:
            df['prev_jockey_id'] = df.groupby('horse_id')['jockey_id'].shift(1)
            df['jockey_change_flag'] = (
                (df['jockey_id'] != df['prev_jockey_id']) & 
                (df['prev_jockey_id'].notna())
            ).astype('int8')
            
            # 一時カラム削除
            del df['prev_jockey_id']
        
        # ------------------------------------------------------
        # 5. 斤量経験（今回斤量 vs 過去最高斤量）
        # ------------------------------------------------------
        if 'impost' in df.columns:
            df['max_impost_before'] = df.groupby('horse_id')['impost'].transform(
                lambda x: x.shift(1).expanding().max()
            ).fillna(df['impost'])
            
            # 今回の斤量が過去最高かどうか
            df['is_career_high_impost'] = (df['impost'] > df['max_impost_before']).astype('int8')
            
            # 一時カラム削除
            del df['max_impost_before']
        
        logger.info("経験特徴量の生成完了")
        return df


import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_training_detail(df: pd.DataFrame) -> pd.DataFrame:
    """
    調教詳細特徴量 (Training Detail) を計算する。
    Loader改修により、jvd_hc (坂路) と jvd_wc (コース) が統合され、
    'training_course_cat' などが利用可能になっている前提。
    
    Features:
    - training_course_cat: 調教コース種別 (1:坂路, 2:W, 3:Poly, etc) from Loader
    - training_intensity_score: (仮) タイムに基づく強度スコア (単純なタイム逆数など)
    - training_acceleration: ラスト1Fの加速 (Avg Lap vs Last 1F)
    """
    logger.info("Computing Training Detail features...")
    
    req_cols = ['race_id', 'horse_number', 'horse_id']
    base_cols = req_cols + ['training_time_4f', 'training_time_3f', 'training_time_last1f']
    
    # Loaderで course_type が追加される予定
    # もし無ければ 0 (Unknown)
    if 'training_course_cat' not in df.columns:
        df['training_course_cat'] = 0
    else:
        df['training_course_cat'] = df['training_course_cat'].fillna(0).astype(int)
        
    df_feat = df[base_cols + ['training_course_cat']].copy()
    
    # 1. Acceleration (Last 1F vs Avg Lap)
    # 4F time -> 4F - 1F = 3F time. 3F / 3 = Avg Lap (prev).
    # Last 1F.
    # Accel = Avg Lap - Last 1F (正なら加速、負なら失速)
    
    # 4Fが無い場合は3Fを使う
    # 時系列: 4F -> 3F -> 1F ではない。 4F全体タイム、3F全体タイム。
    # 4F - 1F = First 3F.
    # First 3F / 3 = Avg First Lap.
    
    # [Fix] 0除算注意
    def calc_accel(row):
        t4 = row['training_time_4f']
        t3 = row['training_time_3f']
        t1 = row['training_time_last1f']
        
        # 4Fがある場合
        if pd.notnull(t4) and t4 > 0 and pd.notnull(t1) and t1 > 0:
            first_part = t4 - t1
            if first_part > 0:
                avg_first = first_part / 3.0
                return avg_first - t1 # Larger is better acceleration
        
        # 3Fがある場合 (4Fなし)
        if pd.notnull(t3) and t3 > 0 and pd.notnull(t1) and t1 > 0:
            first_part = t3 - t1
            if first_part > 0:
                avg_first = first_part / 2.0
                return avg_first - t1
                
        return 0.0
        
    df_feat['training_acceleration'] = df_feat.apply(calc_accel, axis=1)
    
    # 2. Training Intensity Score (Simple Time Proxy)
    # 坂路とウッドで基準が違うが、とりあえず混ぜて正規化前の生タイム評価
    # 速いほど良い -> 1 / time
    # しかし距離が違うので、1Fあたりのタイムに換算?
    # training_time_4f / 4
    
    def calc_intensity(row):
        t4 = row['training_time_4f']
        t3 = row['training_time_3f']
        if pd.notnull(t4) and t4 > 10: # 10s以上 (異常値除外)
            return 40.0 / t4 # 4F avg speed (inv time)
        if pd.notnull(t3) and t3 > 10:
            return 30.0 / t3
        return 0.0
        
    df_feat['training_intensity_score'] = df_feat.apply(calc_intensity, axis=1)
    
    # カテゴリはそのまま
    # training_course_cat
    
    # Return
    ret_cols = ['training_course_cat', 'training_acceleration', 'training_intensity_score']
    # merge keys
    return df_feat[req_cols + ret_cols].copy()

def get_feature_names():
    return ['training_course_cat', 'training_acceleration', 'training_intensity_score']

"""
Rest Pattern Features - 休養・間隔分析

Features:
- rest_success_rate: 過去の休み明け初戦時（60日以上）の好走率
- optimal_rest_diff: 好走時の平均間隔との差
- tataki_effectiveness: 休み明け2戦目での順位向上傾向（叩き良化型？）
- long_rest_flag: 長期休養（180日以上）明け
"""
import pandas as pd
import numpy as np
from typing import List

def compute_rest_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """
    休養パターン特徴量を計算
    
    Args:
        df: race_id, horse_id, horse_number, date, interval_days, rank, interval_category
    
    Returns:
        DataFrame with rest pattern features
    """
    keys = ['race_id', 'horse_number', 'horse_id']
    df = df.copy()
    
    # Sort
    df = df.sort_values(['horse_id', 'date'])
    
    # Ensure interval_days
    if 'interval_days' not in df.columns:
        # Calculate from date if missing, but usually provided by history_stats block
        # Assuming it exists or calculate it
        df['date'] = pd.to_datetime(df['date'])
        df['interval_days'] = df.groupby('horse_id')['date'].diff().dt.days
    
    df['interval_days'] = df['interval_days'].fillna(0)
    
    # Define rest (break) as >= 60 days
    REST_THRESHOLD = 60
    LONG_REST_THRESHOLD = 180
    
    df['is_rest'] = (df['interval_days'] >= REST_THRESHOLD).astype(int)
    df['is_long_rest'] = (df['interval_days'] >= LONG_REST_THRESHOLD).astype(int)
    
    # Win/Top3
    df['is_top3'] = (df['rank'] <= 3).astype(int)
    
    # --- 1. Rest Success Rate ---
    # Win rate when coming from rest
    
    # Only rows where is_rest=1 matters for the denominator
    # We want to know: Of the times this horse raced after rest, how often did it win?
    
    # Isolate rest races
    df['rest_race_top3'] = np.where(df['is_rest'] == 1, df['is_top3'], np.nan)
    
    # Expanding mean of rest_race_top3
    # Shift to prevent leakage
    df['rest_success_rate'] = df.groupby('horse_id')['rest_race_top3'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(0)
    
    # --- 2. Optimal Rest Diff ---
    # Find avg interval for top3 races
    df['top3_interval'] = np.where(df['is_top3'] == 1, df['interval_days'], np.nan)
    
    df['avg_winning_interval'] = df.groupby('horse_id')['top3_interval'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    
    # If no past wins, use career avg or 0? 0 is safer
    df['avg_winning_interval'] = df['avg_winning_interval'].fillna(0)
    
    df['optimal_rest_diff'] = df['interval_days'] - df['avg_winning_interval']
    
    # --- 3. Tataki Effectiveness (2nd race improvement) ---
    # "Tataki" means racing once after rest to sharpen up.
    # Logic: If current race is 2nd race after rest (prev was rest),
    # how much does this horse typically improve from 1st to 2nd?
    
    # Lag variables
    df['prev_is_rest'] = df.groupby('horse_id')['is_rest'].shift(1).fillna(0)
    df['prev_rank'] = df.groupby('horse_id')['rank'].shift(1).fillna(18) # default to poor rank
    
    # Identify if PREVIOUS race was a rest race
    df['is_second_after_rest'] = df['prev_is_rest']
    
    # Calculate rank improvement (Prev - Curr). Positive = Improved.
    df['rank_improvement'] = df['prev_rank'] - df['rank']
    
    # Store improvement for 2nd-after-rest races
    df['tataki_improvement'] = np.where(df['is_second_after_rest'] == 1, df['rank_improvement'], np.nan)
    
    # Expanding mean of improvement
    df['tataki_effectiveness'] = df.groupby('horse_id')['tataki_improvement'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(0)
    
    # --- Output Features ---
    feats = [
        'rest_success_rate',
        'optimal_rest_diff',
        'tataki_effectiveness',
        'is_long_rest'
    ]
    
    available_keys = [k for k in keys if k in df.columns]
    result = df[available_keys + feats].copy()
    
    for f in feats:
        result[f] = result[f].fillna(0)
        
    return result

def get_feature_names() -> List[str]:
    return [
        'rest_success_rate',
        'optimal_rest_diff',
        'tataki_effectiveness',
        'is_long_rest'
    ]

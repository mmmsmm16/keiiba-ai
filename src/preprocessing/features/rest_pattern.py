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
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
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
    grp_h = df.groupby('horse_id')
    df['rest_success_rate'] = grp_h['rest_race_top3'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(0)
    
    # Smoothed + decayed rest success (improved)
    rest_count = grp_h['is_rest'].transform(lambda x: x.cumsum().shift(1)).fillna(0.0)
    rest_top3_sum = grp_h['rest_race_top3'].transform(lambda x: x.fillna(0.0).cumsum().shift(1)).fillna(0.0)
    global_top3 = df['is_top3'].expanding(min_periods=1).mean().shift(1).fillna(0.25)
    prior = 8.0
    df['rest_success_rate_smoothed'] = (rest_top3_sum + prior * global_top3) / (rest_count + prior)
    df['rest_success_rate_decay'] = grp_h['rest_race_top3'].transform(
        lambda x: x.shift(1).ewm(alpha=0.25, adjust=False, min_periods=1).mean()
    ).fillna(df['rest_success_rate_smoothed'])
    
    # --- 2. Optimal Rest Diff ---
    # Find avg interval for top3 races
    df['top3_interval'] = np.where(df['is_top3'] == 1, df['interval_days'], np.nan)
    
    df['avg_winning_interval'] = df.groupby('horse_id')['top3_interval'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    
    # If no past wins, use career avg or 0? 0 is safer
    df['avg_winning_interval'] = df['avg_winning_interval'].fillna(0)
    
    df['optimal_rest_diff'] = df['interval_days'] - df['avg_winning_interval']
    top3_interval_std = grp_h['top3_interval'].transform(
        lambda x: x.expanding().std().shift(1)
    ).fillna(30.0).clip(lower=7.0)
    df['rest_optimality_score'] = np.exp(-np.abs(df['optimal_rest_diff']) / (top3_interval_std + 1e-6))
    
    # --- 3. Tataki Effectiveness (2nd race improvement) ---
    # "Tataki" means racing once after rest to sharpen up.
    # Logic: If current race is 2nd race after rest (prev was rest),
    # how much does this horse typically improve from 1st to 2nd?
    
    # Lag variables
    df['prev_is_rest'] = grp_h['is_rest'].shift(1).fillna(0)
    df['prev_rank'] = grp_h['rank'].shift(1).fillna(18) # default to poor rank
    
    # Identify if PREVIOUS race was a rest race
    df['is_second_after_rest'] = df['prev_is_rest']
    
    # Calculate rank improvement (Prev - Curr). Positive = Improved.
    df['rank_improvement'] = df['prev_rank'] - df['rank']
    
    # Store improvement for 2nd-after-rest races
    df['tataki_improvement'] = np.where(df['is_second_after_rest'] == 1, df['rank_improvement'], np.nan)
    
    # Expanding mean of improvement
    df['tataki_effectiveness'] = grp_h['tataki_improvement'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(0)
    
    # --- 4. Rotation Stress ---
    # Travel proxy from venue transitions (rough regional distance)
    if 'venue' in df.columns:
        region_map = {
            '01': 'N', '02': 'N',
            '03': 'E', '04': 'E', '05': 'E', '06': 'E',
            '07': 'W', '08': 'W', '09': 'W', '10': 'W'
        }
        dist_map = {
            ('E', 'E'): 0.0, ('W', 'W'): 0.0, ('N', 'N'): 0.0,
            ('E', 'W'): 1.0, ('W', 'E'): 1.0,
            ('E', 'N'): 0.7, ('N', 'E'): 0.7,
            ('W', 'N'): 0.8, ('N', 'W'): 0.8
        }
        df['venue_code'] = df['venue'].astype(str).str.zfill(2)
        df['venue_region'] = df['venue_code'].map(region_map).fillna('U')
        df['prev_venue_region'] = grp_h['venue_region'].shift(1).fillna(df['venue_region'])
        df['travel_load'] = df.apply(
            lambda r: dist_map.get((r['prev_venue_region'], r['venue_region']), 0.4 if 'U' in (r['prev_venue_region'], r['venue_region']) else 0.0),
            axis=1
        )
    else:
        df['travel_load'] = 0.0
    
    if 'tataki_count' in df.columns:
        tataki_norm = pd.to_numeric(df['tataki_count'], errors='coerce').fillna(1.0).clip(lower=1.0, upper=6.0) / 6.0
    else:
        tataki_norm = 0.0
    short_turn = np.maximum(0.0, 21.0 - pd.to_numeric(df['interval_days'], errors='coerce').fillna(0.0)) / 21.0
    df['rotation_stress'] = (
        0.35 * df['travel_load'] +
        0.35 * short_turn +
        0.15 * df['is_long_rest'] +
        0.15 * tataki_norm
    )
    
    # --- 5. Horse variant preference / match ---
    if 'track_variant' in df.columns:
        df['track_variant_num'] = pd.to_numeric(df['track_variant'], errors='coerce').fillna(0.0)
        df['variant_weighted_top3'] = df['track_variant_num'] * df['is_top3']
        pref_num = grp_h['variant_weighted_top3'].transform(lambda x: x.cumsum().shift(1)).fillna(0.0)
        pref_den = grp_h['is_top3'].transform(lambda x: x.cumsum().shift(1)).fillna(0.0)
        df['horse_variant_pref'] = np.where(pref_den > 0, pref_num / pref_den, 0.0)
        df['horse_variant_match'] = -np.abs(df['track_variant_num'] - df['horse_variant_pref'])
    else:
        df['horse_variant_pref'] = 0.0
        df['horse_variant_match'] = 0.0
    
    # --- Output Features ---
    feats = [
        'rest_success_rate',
        'rest_success_rate_smoothed',
        'rest_success_rate_decay',
        'optimal_rest_diff',
        'rest_optimality_score',
        'tataki_effectiveness',
        'is_long_rest',
        'rotation_stress',
        'horse_variant_pref',
        'horse_variant_match'
    ]
    
    available_keys = [k for k in keys if k in df.columns]
    result = df[available_keys + feats].copy()
    
    for f in feats:
        result[f] = result[f].fillna(0)
        
    return result

def get_feature_names() -> List[str]:
    return [
        'rest_success_rate',
        'rest_success_rate_smoothed',
        'rest_success_rate_decay',
        'optimal_rest_diff',
        'rest_optimality_score',
        'tataki_effectiveness',
        'is_long_rest',
        'rotation_stress',
        'horse_variant_pref',
        'horse_variant_match'
    ]

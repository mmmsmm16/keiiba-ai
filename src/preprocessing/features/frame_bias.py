"""
Frame Bias Features - 枠順バイアス分析

競馬場・距離・馬場状態によって枠順の有利不利が異なる。
例: 中山芝1200m内枠有利、東京ダート内枠不利

Features:
- frame_number: 枠番 (1-8)
- frame_win_rate_venue_dist: 会場×距離別の枠勝率
- inner_outer_flag: 内枠(1-4) vs 外枠(5-8)
- frame_surface_bias: 芝/ダート別の枠バイアス
"""
import pandas as pd
import numpy as np
from typing import List

def compute_frame_bias(df: pd.DataFrame) -> pd.DataFrame:
    """
    枠順バイアス特徴量を計算
    
    Args:
        df: race_id, horse_number, frame_number, venue, distance, surface, rank を含むDataFrame
    
    Returns:
        DataFrame with frame bias features
    """
    keys = ['race_id', 'horse_number', 'horse_id']  # Include horse_id for pipeline merge
    df = df.copy()
    df = df.sort_values(['date', 'race_id', 'horse_number'])
    
    # Basic frame features
    if 'frame_number' not in df.columns:
        # frame_number is wakuban in the data
        df['frame_number'] = df.get('wakuban', df.get('frame_number', 0))
    
    df['frame_number'] = pd.to_numeric(df['frame_number'], errors='coerce').fillna(0).astype(int)
    
    # Inner/Outer flag (1-4: inner, 5-8: outer)
    df['is_inner_frame'] = (df['frame_number'] <= 4).astype(int)
    df['is_outer_frame'] = (df['frame_number'] > 4).astype(int)
    
    # Create condition keys for venue x distance x surface
    df['venue'] = df['venue'].astype(str)
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce').fillna(0).astype(int)
    df['surface'] = df['surface'].astype(str)
    
    # Distance category for grouping
    df['dist_cat'] = pd.cut(
        df['distance'], 
        bins=[0, 1400, 1800, 2200, 9999],
        labels=['sprint', 'mile', 'middle', 'long']
    ).astype(str)
    
    # Is win
    df['is_win'] = (df['rank'] == 1).astype(int)
    df['is_top3'] = (df['rank'] <= 3).astype(int)
    
    # --- Frame Win Rate by Venue x Distance Category ---
    # Use expanding + shift to prevent leakage
    cond_keys = ['venue', 'dist_cat', 'surface', 'frame_number']
    
    # Sort for proper expanding calculation
    df_sorted = df.sort_values(['date', 'race_id'])
    
    # Calculate expanding win rate per frame condition
    grp = df_sorted.groupby(cond_keys)
    df_sorted['frame_cond_wins'] = grp['is_win'].transform(lambda x: x.expanding().sum().shift(1))
    df_sorted['frame_cond_count'] = grp['is_win'].transform(lambda x: x.expanding().count().shift(1))
    df_sorted['frame_cond_win_rate'] = (df_sorted['frame_cond_wins'] / df_sorted['frame_cond_count']).fillna(0)
    
    # --- Inner Frame Win Rate by Venue x Distance ---
    inner_cond_keys = ['venue', 'dist_cat', 'surface', 'is_inner_frame']
    grp_inner = df_sorted.groupby(inner_cond_keys)
    df_sorted['inner_wins'] = grp_inner['is_win'].transform(lambda x: x.expanding().sum().shift(1))
    df_sorted['inner_count'] = grp_inner['is_win'].transform(lambda x: x.expanding().count().shift(1))
    df_sorted['inner_frame_win_rate'] = (df_sorted['inner_wins'] / df_sorted['inner_count']).fillna(0)
    
    # --- Frame Bias Score (inner advantage) ---
    # Simple bias: frame_number normalized advantage
    # Lower frame_number = higher advantage in some courses
    df_sorted['frame_advantage_score'] = (9 - df_sorted['frame_number']) / 8  # 1 for frame 1, 0 for frame 8
    
    # Features to output
    feats = [
        'frame_number',
        'is_inner_frame', 
        'frame_cond_win_rate',
        'inner_frame_win_rate',
        'frame_advantage_score'
    ]
    
    # Clean up temp columns - ensure all keys exist
    available_keys = [k for k in keys if k in df_sorted.columns]
    result = df_sorted[available_keys + feats].copy()
    
    # Fill NaN
    for f in feats:
        if f in result.columns:
            result[f] = result[f].fillna(0)
    
    return result

def get_feature_names() -> List[str]:
    """Return list of feature names produced by this block."""
    return [
        'frame_number',
        'is_inner_frame',
        'frame_cond_win_rate',
        'inner_frame_win_rate', 
        'frame_advantage_score'
    ]

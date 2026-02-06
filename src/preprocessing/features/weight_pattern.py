"""
Weight Pattern Features - 馬体重分析

Features:
- weight: 馬体重 (Base)
- weight_diff: 体重増減 (Base)
- weight_trend_3: 直近3走の体重トレンド (増加傾向/減少傾向)
- weight_volatility_5: 直近5走の体重変動標準偏差
- weight_season_diff: 前年同時期との体重差
- optimal_weight_diff: 過去ベストパフォーマンス時との体重差
"""
import pandas as pd
import numpy as np
from typing import List

def compute_weight_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """
    馬体重パターン特徴量を計算
    
    Args:
        df: race_id, horse_id, horse_number, date, weight, rank
    
    Returns:
        DataFrame with weight pattern features
    """
    # Required columns
    req_cols = ['race_id', 'horse_id', 'horse_number', 'date', 'weight', 'rank']
    
    # Check if we have required cols
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        # Some might be in different names or need loading
        # For now assume mostly standard names. weight might be bataiju
        if 'weight' not in df.columns and 'bataiju' in df.columns:
            df['weight'] = df['bataiju']
            
    # Filter and sort
    df_proc = df.copy()
    
    # Clean weight
    df_proc['weight'] = pd.to_numeric(df_proc['weight'], errors='coerce')
    # Fill weight with mean? Or forward fill?
    # Group by horse_id and ffill
    df_proc = df_proc.sort_values(['horse_id', 'date'])
    df_proc['weight'] = df_proc.groupby('horse_id')['weight'].ffill()
    
    # --- 1. Weight Volatility (Stability) ---
    # Rolling 5 std
    df_proc['weight_volatility_5'] = df_proc.groupby('horse_id')['weight'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).std()
    ).fillna(0)
    
    # --- 2. Weight Trend (Short term) ---
    # Current weight - Avg of last 3
    df_proc['avg_weight_3'] = df_proc.groupby('horse_id')['weight'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df_proc['weight_trend_3'] = df_proc['weight'] - df_proc['avg_weight_3']
    
    # --- 3. Seasonal Weight Diff ---
    # Compare with weight roughly 365 days ago (+- 60 days)
    # This is tricky with rolling, maybe skip for now or use simple approximation
    # Approximation: Avg weight of races between 300-400 days ago
    # We can use lag features if we had date index, but here we don't.
    # Let's use a simpler proxy: Weight - Career Avg Weight
    df_proc['career_avg_weight'] = df_proc.groupby('horse_id')['weight'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df_proc['weight_vs_avg'] = df_proc['weight'] - df_proc['career_avg_weight']
    
    # --- 4. Optimal Weight Diff ---
    # Find weight when horse won or placed (rank <= 3)
    # If multiple, take mean
    
    # Identify good performance rows
    good_perf = df_proc[df_proc['rank'] <= 3].copy()
    
    # We need expanding optimal weight. 
    # For each row, calculate mean weight of PREVIOUS wins/places
    
    # Add is_good_perf flag
    df_proc['is_good_perf'] = (df_proc['rank'] <= 3).astype(int)
    
    # Mask weights where perf was not good
    df_proc['good_weight'] = np.where(df_proc['is_good_perf'], df_proc['weight'], np.nan)
    
    # Expanding mean of good weights
    df_proc['optimal_weight'] = df_proc.groupby('horse_id')['good_weight'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    
    # Diff
    df_proc['optimal_weight_diff'] = df_proc['weight'] - df_proc['optimal_weight']
    # If no past good perf, fill with 0 (neutral) or maybe NaN?
    # Let's fill with 0 assumption that current weight is okay if unknown
    df_proc['optimal_weight_diff'] = df_proc['optimal_weight_diff'].fillna(0)
    
    # Select output columns
    feats = [
        'weight_volatility_5',
        'weight_trend_3',
        'weight_vs_avg',
        'optimal_weight_diff'
    ]
    
    keys = ['race_id', 'horse_number', 'horse_id'] 
    result = df_proc[keys + feats].copy()
    
    for f in feats:
        result[f] = result[f].fillna(0)
        
    return result

def get_feature_names() -> List[str]:
    return [
        'weight_volatility_5',
        'weight_trend_3',
        'weight_vs_avg',
        'optimal_weight_diff'
    ]

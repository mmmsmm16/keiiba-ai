"""
Corner Dynamics Features - コーナー通過分析

Features:
- position_change_3_4: 3-4コーナーでの位置取り変化（捲り検知）
- corner_advance_score: コーナーで順位を上げた度合い（機動力）
- final_corner_pos_pct: 4コーナー通過順位の相対位置（展開）
- corner_acceleration: 4コーナー位置からの加速力（4角位置 vs 上がり3F）
"""
import pandas as pd
import numpy as np
from typing import List

def compute_corner_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """
    コーナー通過ダイナミクス特徴量を計算
    
    Args:
        df: race_id, horse_id, horse_number, date, corner_1, corner_2, corner_3, corner_4, rank, last_3f_rank, heads_count
    
    Returns:
        DataFrame with corner dynamics features
    """
    keys = ['race_id', 'horse_number', 'horse_id']
    df = df.copy()
    
    # Sort
    df = df.sort_values(['horse_id', 'date'])
    
    # Preprocess corner positions
    # corner_N is often string like "1" or "10" or "4,5" (if multiple passings, usually simplified to last passing)
    # or sometimes missing.
    # We need purely numeric values.
    # Note: JRA-VAN data might have corner like '10(2)' or '3-3-3'. 
    # Assuming loader has cleaned it or we need to clean it. 
    # For now assume numeric or convertible. If it's a passing order string, we take the first number.
    
    def clean_corner_pos(series):
        # Taking the first numeric value if string
        return pd.to_numeric(series.astype(str).str.extract(r'(\d+)')[0], errors='coerce')

    for c in ['corner_3', 'corner_4']:
        if c in df.columns:
            df[f'{c}_clean'] = clean_corner_pos(df[c])
        else:
            df[f'{c}_clean'] = np.nan

    # --- 1. Position Change 3-4 (Makuri/Drop) ---
    # Negative value = Moved forward (e.g. 10 -> 5) -> Good (Makuri)
    # Positive value = Dropped back -> Bad
    # We want "Advance" so (3c - 4c)
    df['position_change_3_4'] = df['corner_3_clean'] - df['corner_4_clean']
    
    # Normalize with heads count if possible, but raw change is also useful.
    # Let's keep raw change but handle NaNs (e.g. only 4 corners race)
    df['position_change_3_4'] = df['position_change_3_4'].fillna(0)
    
    # --- 2. Final Corner Position Pct ---
    # Relative position at 4th corner (Entering straight)
    if 'heads_count' in df.columns:
        df['heads_count'] = pd.to_numeric(df['heads_count'], errors='coerce')
        df['final_corner_pos_pct'] = df['corner_4_clean'] / df['heads_count']
    else:
        # Fallback if no head count (unlikely)
        df['final_corner_pos_pct'] = df['corner_4_clean'] / 14.0 # Avg field size
        
    df['final_corner_pos_pct'] = df['final_corner_pos_pct'].fillna(0.5) # Default mid-pack
    
    # --- 3. Corner Advance Score (Aggregated) ---
    # Average position gain in corners
    # Shifted expanding mean of position_change_3_4
    # Positive = Tendency to advance in corners (Agility)
    df['corner_advance_score'] = df.groupby('horse_id')['position_change_3_4'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(0)
    
    # --- 4. Corner Acceleration (Pos 4c vs Last 3F Rank) ---
    # Correlation between being back and having fast last 3F is normal
    # But if you are forward (low 4c) AND have fast last 3F (low 3f_rank), that's strong.
    # Or if you are back (high 4c) and pass many horses (Rank < 4c).
    
    if 'last_3f_rank' not in df.columns:
        # Create if missing (approx)
        df['last_3f_rank'] = 8 # mid
        
    df['last_3f_rank'] = pd.to_numeric(df['last_3f_rank'], errors='coerce').fillna(8)
    
    # Passes in straight: 4c - Rank
    # Positive = Passed horses in straight
    df['straight_passes'] = df['corner_4_clean'] - df['rank']
    
    df['corner_acceleration'] = df.groupby('horse_id')['straight_passes'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(0)
    
    # --- Output Features ---
    feats = [
        'corner_advance_score',
        'final_corner_pos_pct', # This is current race feature? No, we should use past avg for prediction
        # 'position_change_3_4', # Current race feature -> leakage! Don't use directly.
        'corner_acceleration'
    ]
    
    # Wait, 'final_corner_pos_pct' calculated above is for CURRENT race.
    # We need historical average for prediction features.
    
    # Average Final Corner Positioning (Runstyle proxy)
    df['avg_final_corner_pct'] = df.groupby('horse_id')['final_corner_pos_pct'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(0.5)
    
    # Update feats list to use only historical (lagged) features
    feats = [
        'corner_advance_score',
        'avg_final_corner_pct',
        'corner_acceleration'
    ]
    
    available_keys = [k for k in keys if k in df.columns]
    result = df[available_keys + feats].copy()
    
    for f in feats:
        result[f] = result[f].fillna(0)
        
    return result

def get_feature_names() -> List[str]:
    return [
        'corner_advance_score',
        'avg_final_corner_pct',
        'corner_acceleration'
    ]

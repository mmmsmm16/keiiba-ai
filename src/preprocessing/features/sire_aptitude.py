
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_sire_aptitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block] Sire Aptitude (血統適性)
    - 種牡馬(sire_id)および母父(damsire_id - if available)のコース別・距離別成績を集計する。
    - Target Encodingの一種であるため、Leakage防止（Expanding Mean + Shift）が必須。
    
    Features:
    - sire_course_win_rate: 種牡馬の当該コース(course_id)における累積勝率。
    - sire_dist_win_rate: 種牡馬の当該距離カテゴリ(distance_category)における累積勝率。
    - sire_type_win_rate: 種牡馬の当該芝ダ区分(surface)における累積勝率。
    - (Option) damsire versions if data available. (Assuming we might not have damsire_id easily linked here, 
      but if 'bloodline_stats' block exists, maybe we can access it? 
      Actually, standard JraVanDataLoader provides 'sire_id'. 'damsire_id' might be in bloodline info.
      Let's stick to 'sire_id' which is definitely in columns.)
    """
    logger.info("ブロック計算中: compute_sire_aptitude")
    
    keys = ['race_id', 'horse_number', 'horse_id', 'date']
    
    # Required columns (input)
    # course_id is usually not present, so we construct it.
    req_cols = ['sire_id', 'venue', 'distance', 'surface', 'rank'] 
    
    # Check input cols
    for c in req_cols:
        if c not in df.columns:
            # If rank missing (inference), proceed without target? No, we need it for aggregation.
            # But maybe we are just mapping? No, this block calculates expanding target stats.
            # If training, rank must usually be there.
            pass

    df_sorted = df.sort_values(['date', 'race_id']).copy()
    
    # Create internal course_id
    # Format: Venue_Surface_Dist (e.g. 東京_芝_2400)
    df_sorted['course_id'] = (
        df_sorted['venue'].astype(str) + '_' + 
        df_sorted['surface'].astype(str) + '_' + 
        df_sorted['distance'].astype(str)
    )
    
    # Create Target
    
    # Create Target
    if 'rank' in df_sorted.columns:
        df_sorted['is_win'] = (df_sorted['rank'] == 1).astype(float)
    else:
        df_sorted['is_win'] = np.nan
        
    # Distance Category (Rounded to 100m or specific bins)
    # 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2500, ..
    # Simple binning: round to nearest 100?
    df_sorted['dist_bin'] = (df_sorted['distance'] // 100) * 100
    
    # 1. Sire x Course Win Rate
    # Group by [sire_id, course_id]
    # Expanding Mean
    # Prior (Laplace Smoothing) to handle small samples
    # global_win_rate approx 0.08 (1/12)
    prior = 10.0
    global_rate = 0.08
    
    def smooth_expanding(x):
        cumsum = x.cumsum().shift(1).fillna(0)
        count = x.expanding().count().shift(1).fillna(0)
        return (cumsum + prior * global_rate) / (count + prior)

    # Optimization: Use transform on grouped objects
    # Sire x Course
    # Note: This can be heavy.
    # To speed up, we might concat keys?
    
    # Sire x Course
    df_sorted['sire_course_win_rate'] = df_sorted.groupby(['sire_id', 'course_id'])['is_win'].transform(smooth_expanding)
    
    # Sire x Distance
    df_sorted['sire_dist_win_rate'] = df_sorted.groupby(['sire_id', 'dist_bin'])['is_win'].transform(smooth_expanding)
    
    # Sire x Surface (Turf/Dirt)
    df_sorted['sire_surface_win_rate'] = df_sorted.groupby(['sire_id', 'surface'])['is_win'].transform(smooth_expanding)
    
    # Fill Nans (First time appearance)
    df_sorted['sire_course_win_rate'] = df_sorted['sire_course_win_rate'].fillna(global_rate)
    df_sorted['sire_dist_win_rate'] = df_sorted['sire_dist_win_rate'].fillna(global_rate)
    df_sorted['sire_surface_win_rate'] = df_sorted['sire_surface_win_rate'].fillna(global_rate)
    
    cols = [
        'sire_course_win_rate',
        'sire_dist_win_rate',
        'sire_surface_win_rate'
    ]
    
    return df_sorted[keys + cols].copy()

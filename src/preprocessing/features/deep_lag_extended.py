import pandas as pd
import numpy as np

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 3.8] Deep Lag Extended
    - 既存のDeepLag (Rank, TimeDiff) に加え、条件/斤量/位置取りなどをLag化
    - lag1_impost, lag1_field_size, lag1_3f_time など
    - [追加] lag1/2/3_venue, lag2/3_surface, lag2/3_distance, lag2/3_going_code
    """
    cols = ['race_id', 'horse_number', 'horse_id']
    df_sorted = df.sort_values(['horse_id', 'date']).copy()
    grp = df_sorted.groupby('horse_id')
    
    out_cols = []
    
    # -------------------------------------------------------------------------
    # Lag 1 Targets (Single Lag)
    # -------------------------------------------------------------------------
    lag1_targets = [
        ('impost', 'lag1_impost'),
        ('field_size', 'lag1_field_size'),
        ('last_3f', 'lag1_last_3f'),
        ('first_corner_rank', 'lag1_first_corner_rank'),
        ('surface', 'lag1_surface'),
        ('distance', 'lag1_distance'),
        ('going_code', 'lag1_going_code'),
        ('venue', 'lag1_venue'),  # NEW: 前走の競馬場
    ]
    
    for src, dst in lag1_targets:
        if src in df_sorted.columns:
            df_sorted[dst] = grp[src].shift(1)
            out_cols.append(dst)

    # -------------------------------------------------------------------------
    # Lag 2 Targets
    # -------------------------------------------------------------------------
    lag2_targets = [
        ('venue', 'lag2_venue'),
        ('surface', 'lag2_surface'),
        ('distance', 'lag2_distance'),
        ('going_code', 'lag2_going_code'),
        ('impost', 'lag2_impost'),
        ('field_size', 'lag2_field_size'),
    ]
    
    for src, dst in lag2_targets:
        if src in df_sorted.columns:
            df_sorted[dst] = grp[src].shift(2)
            out_cols.append(dst)

    # -------------------------------------------------------------------------
    # Lag 3 Targets
    # -------------------------------------------------------------------------
    lag3_targets = [
        ('venue', 'lag3_venue'),
        ('surface', 'lag3_surface'),
        ('distance', 'lag3_distance'),
        ('going_code', 'lag3_going_code'),
    ]
    
    for src, dst in lag3_targets:
        if src in df_sorted.columns:
            df_sorted[dst] = grp[src].shift(3)
            out_cols.append(dst)
            
    # -------------------------------------------------------------------------
    # Extra: Impost Diff (今回 - 前走)
    # -------------------------------------------------------------------------
    if 'impost' in df_sorted.columns and 'lag1_impost' in df_sorted.columns:
        df_sorted['impost_diff_prev'] = df_sorted['impost'] - df_sorted['lag1_impost']
        out_cols.append('impost_diff_prev')

    return df_sorted[cols + out_cols].copy()


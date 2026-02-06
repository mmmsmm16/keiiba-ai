import pandas as pd
import numpy as np

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 3.13] 相対化拡張 (Relative Expansion)
    - 新規追加した特徴量 (Elo, Form, TrainerStats) のレース内相対化
    - Z-score, RankPct, Diff
    """
    cols = ['race_id', 'horse_number', 'horse_id']
    df_sorted = df.copy()
    out_cols = []
    
    # Target columns to relativize
    targets = [
        'horse_elo',
        'speed_index_ewm_5', # from form_trend
        'rank_slope_5',
        'jockey_win_rate_30d', # from stable_form
        'trainer_win_rate_30d',
        'track_variant' # Wait, track_variant is constant per race (mostly). Relative is 0. Skip.
    ]
    
    grp = df_sorted.groupby('race_id')
    
    for col in targets:
        if col not in df_sorted.columns: continue
        
        # Mean, Std
        r_mean = grp[col].transform('mean')
        r_std = grp[col].transform('std').fillna(1.0).replace(0, 1.0)
        
        # Z
        z_col = f"relative_{col}_z"
        df_sorted[z_col] = ((df_sorted[col] - r_mean) / r_std).fillna(0)
        out_cols.append(z_col)
        
        # Rank Pct
        p_col = f"relative_{col}_pct"
        df_sorted[p_col] = grp[col].transform(lambda x: x.rank(pct=True)).fillna(0.5)
        out_cols.append(p_col)
        
        # Gap to Top (for Elo specifically)
        if col == 'horse_elo':
            r_max = grp[col].transform('max')
            df_sorted['relative_elo_gap'] = r_max - df_sorted[col]
            out_cols.append('relative_elo_gap')

    return df_sorted[cols + out_cols].copy()

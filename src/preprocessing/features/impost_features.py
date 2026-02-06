import pandas as pd
import numpy as np

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 3.2] 斤量変動 (Impost Features)
    - impost_change: 今回斤量 - 前走斤量
    - impost_change_abs: 絶対値
    """
    cols = ['race_id', 'horse_number', 'horse_id']
    req = ['impost', 'date']
    
    # Check
    if not all(c in df.columns for c in req):
        return pd.DataFrame()

    df_sorted = df.sort_values(['horse_id', 'date']).copy()
    grp = df_sorted.groupby('horse_id')

    # Lag1 Impost
    df_sorted['lag1_impost'] = grp['impost'].shift(1)
    
    # Change
    df_sorted['impost_change'] = df_sorted['impost'] - df_sorted['lag1_impost']
    # FillNA with 0 (First run or data missing)
    df_sorted['impost_change'] = df_sorted['impost_change'].fillna(0)
    
    # Abs
    df_sorted['impost_change_abs'] = df_sorted['impost_change'].abs()

    out_cols = ['impost_change', 'impost_change_abs']
    return df_sorted[cols + out_cols].copy()

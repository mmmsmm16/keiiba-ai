import pandas as pd
import numpy as np

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 3.10] 馬イベント (Horse Events)
    - 転厩 (trainer_change)
    - 去勢直後 (gelding_after)
    """
    cols = ['race_id', 'horse_number', 'horse_id']
    df_sorted = df.sort_values(['horse_id', 'date']).copy()
    grp = df_sorted.groupby('horse_id')
    
    out_cols = []
    
    # 1. Trainer Change
    if 'trainer_id' in df_sorted.columns:
        df_sorted['prev_trainer'] = grp['trainer_id'].shift(1)
        # Change detected if prev exists and different
        df_sorted['trainer_change'] = (df_sorted['trainer_id'] != df_sorted['prev_trainer']) & df_sorted['prev_trainer'].notnull()
        df_sorted['trainer_change'] = df_sorted['trainer_change'].astype(int)
        out_cols.append('trainer_change')
        
    # 2. Gelding Event
    # Check sex change
    if 'sex' in df_sorted.columns:
        df_sorted['prev_sex'] = grp['sex'].shift(1)
        # 'セ' means Gelding. '牡' means Male.
        # Change Male -> Gelding
        df_sorted['is_gelding_event'] = (df_sorted['prev_sex'] == '牡') & (df_sorted['sex'] == 'セ')
        df_sorted['first_run_after_gelding'] = df_sorted['is_gelding_event'].astype(int)
        out_cols.append('first_run_after_gelding')
        
    return df_sorted[cols + out_cols].copy()

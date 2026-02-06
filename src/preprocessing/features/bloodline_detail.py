
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_nicks_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block] ニックス/血統詳細データ (Nicks Stats)
    - Sire x BMS (母父) の組み合わせ成績 (Nicks)
    - BMS (母父) 単体の成績
    - リーク防止: 過去のレース結果のみを集計 (shift(1))
    """
    logger.info("ブロック計算中: bloodline_detail (Nicks)")
    
    # 必要なカラムの確認
    req_cols = ['race_id', 'horse_number', 'horse_id', 'date', 'sire_id', 'bms_id', 'rank', 'surface']
    for c in req_cols:
        if c not in df.columns:
            if c == 'bms_id':
                 logger.warning(f"bloodline_detail: {c} not found. Skipping Nicks computation.")
                 keys = ['race_id', 'horse_number', 'horse_id']
                 return df[keys].copy()
            raise ValueError(f"bloodline_detail requires column {c}.")

    df_sorted = df[req_cols].copy()
    df_sorted['date'] = pd.to_datetime(df_sorted['date'])
    # 時系列ソート (全体)
    # GroupByのtransform用にはIDソートも必要だが、ここでは「ある配合の過去成績」なので、
    # ペアごとに日付順に並んでいれば良い。
    df_sorted = df_sorted.sort_values(['sire_id', 'bms_id', 'date'])
    
    # ターゲット作成
    df_sorted['is_win'] = (df_sorted['rank'] == 1).astype(int)
    df_sorted['is_top3'] = (df_sorted['rank'] <= 3).astype(int)
    
    # --- 1. Nicks Stats (Sire x BMS) ---
    logger.info("  Computing Nicks (Sire x BMS) stats...")
    
    grp_nicks = df_sorted.groupby(['sire_id', 'bms_id'])
    
    # Vectorized Calculation (cumsum - current)
    # cumcount() starts from 0, so it represents count "before current" effectively?
    # No, cumcount() for 1st element is 0. 2nd is 1.
    # So it IS the count of previous records. Perfect.
    df_sorted['nicks_count'] = grp_nicks.cumcount()
    
    # For sums (wins), cumsum() includes current. Subtract current to get "previous sum".
    # This avoids slow lambda shift.
    nicks_win_inc = grp_nicks['is_win'].cumsum()
    df_sorted['nicks_win_sum'] = nicks_win_inc - df_sorted['is_win']
    
    nicks_top3_inc = grp_nicks['is_top3'].cumsum()
    df_sorted['nicks_top3_sum'] = nicks_top3_inc - df_sorted['is_top3']
    
    # Rates
    def safe_div(a, b):
        return np.where(b > 0, a / b, 0.0)
        
    df_sorted['nicks_win_rate'] = safe_div(df_sorted['nicks_win_sum'], df_sorted['nicks_count'])
    df_sorted['nicks_top3_rate'] = safe_div(df_sorted['nicks_top3_sum'], df_sorted['nicks_count'])
    
    # --- 2. BMS Stats (Broodmare Sire only) ---
    logger.info("  Computing BMS stats...")
    # ソート順変更 - Must resort by BMS, Date because grouping changed
    df_sorted = df_sorted.sort_values(['bms_id', 'date'])
    grp_bms = df_sorted.groupby('bms_id')
    
    df_sorted['bms_count'] = grp_bms.cumcount()
    
    bms_win_inc = grp_bms['is_win'].cumsum()
    df_sorted['bms_win_sum'] = bms_win_inc - df_sorted['is_win']
    
    bms_top3_inc = grp_bms['is_top3'].cumsum()
    df_sorted['bms_top3_sum'] = bms_top3_inc - df_sorted['is_top3']
    
    df_sorted['bms_win_rate'] = safe_div(df_sorted['bms_win_sum'], df_sorted['bms_count'])
    df_sorted['bms_top3_rate'] = safe_div(df_sorted['bms_top3_sum'], df_sorted['bms_count'])
    
    # Fill Low Sample Nicks with Sire Stats? 
    # Or just return 0. (Let model handle it)
    # Fillna just in case
    cols_to_fill = [
        'nicks_count', 'nicks_win_rate', 'nicks_top3_rate',
        'bms_count', 'bms_win_rate', 'bms_top3_rate'
    ]
    df_sorted[cols_to_fill] = df_sorted[cols_to_fill].fillna(0)
    
    feats = cols_to_fill
    keys = ['race_id', 'horse_number', 'horse_id'] 
    
    return df_sorted[keys + feats].copy()

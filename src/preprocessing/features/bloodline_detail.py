
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
    opt_cols = ['distance', 'going_code', 'state']
    for c in req_cols:
        if c not in df.columns:
            if c == 'bms_id':
                 logger.warning(f"bloodline_detail: {c} not found. Skipping Nicks computation.")
                 keys = ['race_id', 'horse_number', 'horse_id']
                 return df[keys].copy()
            raise ValueError(f"bloodline_detail requires column {c}.")

    use_cols = req_cols + [c for c in opt_cols if c in df.columns]
    df_sorted = df[use_cols].copy()
    df_sorted['date'] = pd.to_datetime(df_sorted['date'])
    # 全体を日付順にしておくことで、groupby cumsum/cumcount が時系列順になる
    df_sorted = df_sorted.sort_values(['date', 'race_id', 'horse_number'])
    
    # ターゲット作成
    df_sorted['is_win'] = (df_sorted['rank'] == 1).astype(int)
    df_sorted['is_top3'] = (df_sorted['rank'] <= 3).astype(int)

    # 文脈キー
    if 'distance' in df_sorted.columns:
        df_sorted['dist_bin'] = (pd.to_numeric(df_sorted['distance'], errors='coerce').fillna(0) // 200) * 200
    else:
        df_sorted['dist_bin'] = -1
    if 'going_code' in df_sorted.columns:
        g = pd.to_numeric(df_sorted['going_code'], errors='coerce').fillna(0).astype(int)
        df_sorted['going_group'] = np.where(g >= 3, 1, 0)
    elif 'state' in df_sorted.columns:
        heavy_codes = {'重', '不良', '03', '04', '3', '4', 3, 4}
        df_sorted['going_group'] = df_sorted['state'].isin(heavy_codes).astype(int)
    else:
        df_sorted['going_group'] = 0
    
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
    grp_bms = df_sorted.groupby('bms_id')
    
    df_sorted['bms_count'] = grp_bms.cumcount()
    
    bms_win_inc = grp_bms['is_win'].cumsum()
    df_sorted['bms_win_sum'] = bms_win_inc - df_sorted['is_win']
    
    bms_top3_inc = grp_bms['is_top3'].cumsum()
    df_sorted['bms_top3_sum'] = bms_top3_inc - df_sorted['is_top3']
    
    df_sorted['bms_win_rate'] = safe_div(df_sorted['bms_win_sum'], df_sorted['bms_count'])
    df_sorted['bms_top3_rate'] = safe_div(df_sorted['bms_top3_sum'], df_sorted['bms_count'])

    # --- 3. Contextual Nicks + Hierarchical Smoothing ---
    # context = surface x dist_bin x going_group
    logger.info("  Computing contextual nicks with hierarchical smoothing...")
    nicks_ctx_cols = ['sire_id', 'bms_id', 'surface', 'dist_bin', 'going_group']
    sire_ctx_cols = ['sire_id', 'surface', 'dist_bin', 'going_group']
    bms_ctx_cols = ['bms_id', 'surface', 'dist_bin', 'going_group']

    grp_nicks_ctx = df_sorted.groupby(nicks_ctx_cols)
    df_sorted['nicks_ctx_count'] = grp_nicks_ctx.cumcount()
    df_sorted['nicks_ctx_win_sum'] = grp_nicks_ctx['is_win'].cumsum() - df_sorted['is_win']
    df_sorted['nicks_ctx_top3_sum'] = grp_nicks_ctx['is_top3'].cumsum() - df_sorted['is_top3']

    grp_sire_ctx = df_sorted.groupby(sire_ctx_cols)
    df_sorted['sire_ctx_count'] = grp_sire_ctx.cumcount()
    df_sorted['sire_ctx_win_sum'] = grp_sire_ctx['is_win'].cumsum() - df_sorted['is_win']
    df_sorted['sire_ctx_top3_sum'] = grp_sire_ctx['is_top3'].cumsum() - df_sorted['is_top3']

    grp_bms_ctx = df_sorted.groupby(bms_ctx_cols)
    df_sorted['bms_ctx_count'] = grp_bms_ctx.cumcount()
    df_sorted['bms_ctx_win_sum'] = grp_bms_ctx['is_win'].cumsum() - df_sorted['is_win']
    df_sorted['bms_ctx_top3_sum'] = grp_bms_ctx['is_top3'].cumsum() - df_sorted['is_top3']

    df_sorted['sire_ctx_win_rate'] = safe_div(df_sorted['sire_ctx_win_sum'], df_sorted['sire_ctx_count'])
    df_sorted['sire_ctx_top3_rate'] = safe_div(df_sorted['sire_ctx_top3_sum'], df_sorted['sire_ctx_count'])
    df_sorted['bms_ctx_win_rate'] = safe_div(df_sorted['bms_ctx_win_sum'], df_sorted['bms_ctx_count'])
    df_sorted['bms_ctx_top3_rate'] = safe_div(df_sorted['bms_ctx_top3_sum'], df_sorted['bms_ctx_count'])

    # Leakage-safe global prior (past-only expanding)
    df_sorted['global_win_prior'] = (
        df_sorted['is_win'].expanding(min_periods=1).mean().shift(1).fillna(0.08)
    )
    df_sorted['global_top3_prior'] = (
        df_sorted['is_top3'].expanding(min_periods=1).mean().shift(1).fillna(0.25)
    )

    alpha = 3.0  # sire prior strength
    beta = 2.0   # bms prior strength
    gamma = 8.0  # global prior strength
    denom = df_sorted['nicks_ctx_count'] + alpha + beta + gamma

    df_sorted['nicks_ctx_win_rate_hier'] = (
        df_sorted['nicks_ctx_win_sum']
        + alpha * df_sorted['sire_ctx_win_rate']
        + beta * df_sorted['bms_ctx_win_rate']
        + gamma * df_sorted['global_win_prior']
    ) / denom

    df_sorted['nicks_ctx_top3_rate_hier'] = (
        df_sorted['nicks_ctx_top3_sum']
        + alpha * df_sorted['sire_ctx_top3_rate']
        + beta * df_sorted['bms_ctx_top3_rate']
        + gamma * df_sorted['global_top3_prior']
    ) / denom

    # --- 4. Sire reproducibility (stability) ---
    logger.info("  Computing sire reproducibility stats...")
    grp_sire = df_sorted.groupby('sire_id')
    df_sorted['sire_top3_std_50'] = grp_sire['is_top3'].transform(
        lambda x: x.shift(1).rolling(50, min_periods=5).std()
    )
    q75 = grp_sire['is_top3'].transform(
        lambda x: x.shift(1).rolling(50, min_periods=8).quantile(0.75)
    )
    q25 = grp_sire['is_top3'].transform(
        lambda x: x.shift(1).rolling(50, min_periods=8).quantile(0.25)
    )
    df_sorted['sire_top3_iqr_50'] = q75 - q25
    
    # Fill Low Sample Nicks with Sire Stats? 
    # Or just return 0. (Let model handle it)
    # Fillna just in case
    cols_to_fill = [
        'nicks_count', 'nicks_win_rate', 'nicks_top3_rate',
        'bms_count', 'bms_win_rate', 'bms_top3_rate',
        'nicks_ctx_count', 'nicks_ctx_win_rate_hier', 'nicks_ctx_top3_rate_hier',
        'sire_top3_std_50', 'sire_top3_iqr_50'
    ]
    df_sorted[cols_to_fill] = df_sorted[cols_to_fill].fillna(0)
    
    feats = cols_to_fill
    keys = ['race_id', 'horse_number', 'horse_id'] 
    
    return df_sorted[keys + feats].copy()

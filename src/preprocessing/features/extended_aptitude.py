
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_extended_aptitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block] Extended Aptitude (適性拡張)
    - Sire (種牡馬) と Jockey (騎手) のコース物理適性を計算する。
    - 初コースの馬や、キャリアの浅い馬に対する補完を目的とする。
    
    Targets:
    1. Sire x Rotation/Straight/Slope
    2. Jockey x Rotation/Straight/Slope
    """
    logger.info("ブロック計算中: compute_extended_aptitude")
    
    # Check Required Columns
    # sire_id might be named slightly differently depending on raw data, but standard is 'sire_id' or 'bloodline_id'?
    # Usually in JraVanDataLoader we have 'sire_id'. If not, we check 'peds_0' (Father).
    
    cols = df.columns
    req_cols = ['horse_id', 'date', 'race_id', 'venue', 'rank', 'jockey_id']
    
    # Check for Sire ID
    sire_col = 'sire_id'
    if sire_col not in cols:
        # Try finding peds_0 or similar if available, otherwise fail gracefully
        if 'peds_0' in cols:
            sire_col = 'peds_0'
        else:
            logger.warning("Sire ID not found. Extended aptitude (Sire) will be skipped/empty.")
            sire_col = None
            
    df_sorted = df.sort_values(['date', 'race_id']).copy() # Sort by date globally for correct grouped expanding
    # Note: For efficient grouped transform, we usually sort by [GroupKey, Date].
    # But here we have multiple group keys (Sire, Jockey).
    # We will sort inside the loop or rely on global sort if stable?
    # Safer to sort by [GroupKey, Date] for each operation, or just use global date sort if GroupBy preserves order.
    # Pandas GroupBy usually preserves order if sort=False? No, better be explicit.
    
    if not np.issubdtype(df_sorted['date'].dtype, np.datetime64):
        df_sorted['date'] = pd.to_datetime(df_sorted['date'])

    # Ensure target
    if 'is_win' not in df_sorted.columns:
         df_sorted['rank_num'] = pd.to_numeric(df_sorted['rank'], errors='coerce')
         df_sorted['is_win'] = (df_sorted['rank_num'] == 1).astype(int)
         df_sorted['is_top3'] = (df_sorted['rank_num'] <= 3).astype(int)

    # --- 1. Define Course Attributes (Duplicate logic from course_aptitude for self-containment) ---
    left_venues = ['東京', '新潟', '中京']
    major_venues = ['東京', '新潟', '中京', '京都', '阪神']
    slope_venues = ['中山', '阪神', '東京', '中京']
    
    def get_attr_rotation(v):
        return 'Left' if v in left_venues else 'Right'
    def get_attr_straight(v):
        return 'Major' if v in major_venues else 'Local'
    def get_attr_slope(v):
        return 'Slope' if v in slope_venues else 'Flat'
        
    df_sorted['venue_str'] = df_sorted['venue'].astype(str)
    df_sorted['attr_rot'] = df_sorted['venue_str'].apply(get_attr_rotation)
    df_sorted['attr_str'] = df_sorted['venue_str'].apply(get_attr_straight)
    df_sorted['attr_slp'] = df_sorted['venue_str'].apply(get_attr_slope)
    
    # --- 2. Calculate Extended Aptitude ---
    
    # Helper
    def calc_group_stat(d, grp_keys, target_col, prefix):
        # Sort by keys + date to ensure correct shifting
        # To avoid massive re-sorting, we can try to rely on global date sort?
        # But safest is explicit. To avoid overhead, maybe just sort once by date?
        # If d is sorted by date, d.groupby(keys) yields groups in order of appearance? No.
        # But within group, is order preserved? Yes, if sort=False (default is True for groupby keys).
        # Actually standard groupby sorts keys. But within group, rows remain in original relative order? Yes.
        # So if df_sorted is sorted by DATE, then within each group, rows are sorted by DATE.
        
        # Calculate expanding mean
        # shift(1) to avoid leakage
        res = d.groupby(grp_keys)[target_col].transform(lambda x: x.expanding().mean().shift(1)).fillna(0)
        return res
        
    # 2.1 Sire Aptitude
    if sire_col:
        logger.info(f"Computing Sire Aptitude using {sire_col}...")
        df_sorted['sire_apt_rot'] = calc_group_stat(df_sorted, [sire_col, 'attr_rot'], 'is_win', 'sire')
        df_sorted['sire_apt_str'] = calc_group_stat(df_sorted, [sire_col, 'attr_str'], 'is_win', 'sire')
        df_sorted['sire_apt_slp'] = calc_group_stat(df_sorted, [sire_col, 'attr_slp'], 'is_win', 'sire')
    else:
        df_sorted['sire_apt_rot'] = 0.0
        df_sorted['sire_apt_str'] = 0.0
        df_sorted['sire_apt_slp'] = 0.0

    # 2.2 Jockey Aptitude
    logger.info("Computing Jockey Aptitude...")
    df_sorted['jockey_apt_rot'] = calc_group_stat(df_sorted, ['jockey_id', 'attr_rot'], 'is_win', 'jockey')
    df_sorted['jockey_apt_str'] = calc_group_stat(df_sorted, ['jockey_id', 'attr_str'], 'is_win', 'jockey')
    df_sorted['jockey_apt_slp'] = calc_group_stat(df_sorted, ['jockey_id', 'attr_slp'], 'is_win', 'jockey')
    
    # We could calculate Top3 rate as well if needed, but Win rate is proxy for "Wins".
    # Let's add Top3 for "Stability" proxy? Maybe just Win for "Aptitude" impact.
    # Actually, Top3 is often more stable for stats. Let's add Top3 as well.
    
    if sire_col:
        df_sorted['sire_apt_rot_top3'] = calc_group_stat(df_sorted, [sire_col, 'attr_rot'], 'is_top3', 'sire')
        df_sorted['sire_apt_str_top3'] = calc_group_stat(df_sorted, [sire_col, 'attr_str'], 'is_top3', 'sire')
        df_sorted['sire_apt_slp_top3'] = calc_group_stat(df_sorted, [sire_col, 'attr_slp'], 'is_top3', 'sire')
    else:
        df_sorted['sire_apt_rot_top3'] = 0.0
        df_sorted['sire_apt_str_top3'] = 0.0
        df_sorted['sire_apt_slp_top3'] = 0.0
        
    df_sorted['jockey_apt_rot_top3'] = calc_group_stat(df_sorted, ['jockey_id', 'attr_rot'], 'is_top3', 'jockey')
    df_sorted['jockey_apt_str_top3'] = calc_group_stat(df_sorted, ['jockey_id', 'attr_str'], 'is_top3', 'jockey')
    df_sorted['jockey_apt_slp_top3'] = calc_group_stat(df_sorted, ['jockey_id', 'attr_slp'], 'is_top3', 'jockey')

    # Return
    feats = [
        'sire_apt_rot', 'sire_apt_str', 'sire_apt_slp',
        'jockey_apt_rot', 'jockey_apt_str', 'jockey_apt_slp',
        'sire_apt_rot_top3', 'sire_apt_str_top3', 'sire_apt_slp_top3',
        'jockey_apt_rot_top3', 'jockey_apt_str_top3', 'jockey_apt_slp_top3'
    ]
    keys = ['race_id', 'horse_number', 'horse_id']
    
    # Re-sort to original order? Not strictly necessary as we return subset, but usually nice.
    # The caller usually merges on keys.
    return df_sorted[keys + feats].copy()

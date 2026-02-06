
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_course_aptitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block] Course Aptitude (コース物理適性)
    - 競馬場名だけでなく、物理特性（回り、直線、坂）への適性を計算する。
    - 初コースや開催替わりでの予測精度向上を目指す。
    
    Attributes:
    1. Rotation (Right/Left)
    2. Straight Type (Major/Local) - 直線の長さ・広さ
    3. Slope Type (Slope/Flat) - ゴール前の急坂有無
    """
    logger.info("ブロック計算中: compute_course_aptitude")
    
    # 必要なカラム
    req_cols = ['horse_id', 'date', 'race_id', 'venue', 'rank', 'is_win'] # is_win might not exist
    
    df_sorted = df.sort_values(['horse_id', 'date']).copy()
    if not np.issubdtype(df_sorted['date'].dtype, np.datetime64):
        df_sorted['date'] = pd.to_datetime(df_sorted['date'])

    # Ensure target
    if 'is_win' not in df_sorted.columns and 'rank' in df_sorted.columns:
         df_sorted['rank_num'] = pd.to_numeric(df_sorted['rank'], errors='coerce')
         df_sorted['is_win'] = (df_sorted['rank_num'] == 1).astype(int)
         df_sorted['is_top3'] = (df_sorted['rank_num'] <= 3).astype(int)
    elif 'is_win' in df_sorted.columns:
         if 'is_top3' not in df_sorted.columns:
             df_sorted['rank_num'] = pd.to_numeric(df_sorted['rank'], errors='coerce')
             df_sorted['is_top3'] = (df_sorted['rank_num'] <= 3).astype(int)
    else:
        # Should not happen
        return df[['race_id', 'horse_number', 'horse_id']]

    # --- 1. Define Course Attributes ---
    # マッピング定義
    # Venue ID or Name? Usually Name in 'venue' col (e.g. "東京", "中山").
    # If codes provided, map codes. Assuming standard JRA names.
    
    # Rotation: Left=東京(05), 新潟(04), 中京(07). Others Right.
    left_venues = ['東京', '新潟', '中京']
    
    # Straight/Track Type: 
    # Major (Long/Wide): 東京, 新潟, 中京, 京都, 阪神
    # Local (Short/Tight): 中山, 札幌, 函館, 福島, 小倉
    # Note: Nakayama has short straight.
    major_venues = ['東京', '新潟', '中京', '京都', '阪神']
    
    # Slope (Finish Line Hill):
    # Slope: 中山, 阪神, 東京, 中京
    # Flat: 京都, 新潟, 札幌, 函館, 福島, 小倉
    slope_venues = ['中山', '阪神', '東京', '中京']
    
    def get_attr_rotation(v):
        if v in left_venues: return 'Left'
        return 'Right'
        
    def get_attr_straight(v):
        if v in major_venues: return 'Major'
        return 'Local'
        
    def get_attr_slope(v):
        if v in slope_venues: return 'Slope'
        return 'Flat'
        
    # Apply Mappings
    # venue column might be category
    df_sorted['venue_str'] = df_sorted['venue'].astype(str)
    
    df_sorted['attr_rot'] = df_sorted['venue_str'].apply(get_attr_rotation)
    df_sorted['attr_str'] = df_sorted['venue_str'].apply(get_attr_straight)
    df_sorted['attr_slp'] = df_sorted['venue_str'].apply(get_attr_slope)
    
    # --- 2. Calculate Aptitude (Expanding Stats) ---
    # GroupBy Horse + Attribute -> Shift(1) -> Expanding Mean
    
    # Rotation Aptitude (Right vs Left)
    # We want "Win Rate in Right" vs "Win Rate in Left"
    # And then extract the one matching current race.
    
    # Helper for transform
    def calc_stat(d, group_cols, target_col):
        # return expanding mean shifted
        return d.groupby(group_cols)[target_col].transform(lambda x: x.expanding().mean().shift(1)).fillna(0)
    
    # 2.1 Rotation
    df_sorted['apt_rot_win_rate'] = calc_stat(df_sorted, ['horse_id', 'attr_rot'], 'is_win')
    df_sorted['apt_rot_top3_rate'] = calc_stat(df_sorted, ['horse_id', 'attr_rot'], 'is_top3')
    
    # 2.2 Straight Type
    df_sorted['apt_str_win_rate'] = calc_stat(df_sorted, ['horse_id', 'attr_str'], 'is_win')
    df_sorted['apt_str_top3_rate'] = calc_stat(df_sorted, ['horse_id', 'attr_str'], 'is_top3')
    
    # 2.3 Slope Type
    df_sorted['apt_slp_win_rate'] = calc_stat(df_sorted, ['horse_id', 'attr_slp'], 'is_win')
    df_sorted['apt_slp_top3_rate'] = calc_stat(df_sorted, ['horse_id', 'attr_slp'], 'is_top3')
    
    # 2.4 Race Count per Attribute (Experience)
    # Experience helps model decide confidence
    df_sorted['apt_rot_count'] = df_sorted.groupby(['horse_id', 'attr_rot']).cumcount()
    df_sorted['apt_str_count'] = df_sorted.groupby(['horse_id', 'attr_str']).cumcount()
    df_sorted['apt_slp_count'] = df_sorted.groupby(['horse_id', 'attr_slp']).cumcount()
    
    # --- 3. First Time Flags ---
    # 初めての条件か？ (Count == 0)
    df_sorted['is_first_rot'] = (df_sorted['apt_rot_count'] == 0).astype(int)
    df_sorted['is_first_str'] = (df_sorted['apt_str_count'] == 0).astype(int)
    df_sorted['is_first_slp'] = (df_sorted['apt_slp_count'] == 0).astype(int)

    # 抽出
    feats = [
        'apt_rot_win_rate', 'apt_rot_top3_rate', 'apt_rot_count', 'is_first_rot',
        'apt_str_win_rate', 'apt_str_top3_rate', 'apt_str_count', 'is_first_str',
        'apt_slp_win_rate', 'apt_slp_top3_rate', 'apt_slp_count', 'is_first_slp'
    ]
    keys = ['race_id', 'horse_number', 'horse_id']
    
    return df_sorted[keys + feats].copy()

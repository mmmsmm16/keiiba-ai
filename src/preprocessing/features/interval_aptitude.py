
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_interval_aptitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block] Interval Aptitude (間隔適性・鮮度・鉄砲)
    - 馬ごとの「レース間隔」に対する適性を特徴量化する。
    - 一般的な「休み明け」フラグだけでなく、「休み明けが得意か？（鉄砲実績）」をスコア化。
    - 「叩き良化型」か「休み明け駆け型」かを判別する。

    Features:
    - current_interval: 今回の間隔（日数）
    - interval_type: Short / Middle / Long
    - apt_interval_win: 今回のinterval_typeにおける過去勝率
    - apt_interval_top3: 今回のinterval_typeにおける過去複勝率
    - is_first_interval_type: 今回のタイプが初めてか
    - tataki_count: 長期休養(Long)明けからの戦数 (1=鉄砲, 2=叩き2戦目...)
    """
    logger.info("ブロック計算中: compute_interval_aptitude")
    
    req_cols = ['race_id', 'horse_number', 'horse_id', 'date', 'rank']
    # Check if necessary columns exist
    if 'date' not in df.columns:
        return df[['race_id', 'horse_number']].copy()

    df_sorted = df.sort_values(['horse_id', 'date']).copy()
    if not np.issubdtype(df_sorted['date'].dtype, np.datetime64):
        df_sorted['date'] = pd.to_datetime(df_sorted['date'])
        
    # Ensure targets
    if 'is_win' not in df_sorted.columns:
        # Try to create from rank
        if 'rank' in df_sorted.columns:
            # Handle string rank if necessary, but usually pipeline does cast
            rank_num = pd.to_numeric(df_sorted['rank'], errors='coerce')
            df_sorted['is_win'] = (rank_num == 1).astype(int)
            df_sorted['is_top3'] = (rank_num <= 3).astype(int)
        else:
            # Cannot compute stats without target
            logger.warning("No rank or target columns found. Returning empty features.")
            return df[['race_id', 'horse_number', 'horse_id']].copy()

    # 1. Calculate Interval
    # shift(1) date
    df_sorted['prev_date'] = df_sorted.groupby('horse_id')['date'].shift(1)
    df_sorted['interval_days'] = (df_sorted['date'] - df_sorted['prev_date']).dt.days
    df_sorted['interval_days'] = df_sorted['interval_days'].fillna(999) # First run -> 999
    
    # 2. Define Interval Types
    # Short: <= 28 (4 weeks) - Rento to Ch3
    # Middle: 29 - 60 (4-8 weeks)
    # Long: > 60 (2 months+) - Yasumi-ake
    
    def get_interval_type(d):
        if d > 1800: return 'New' # Assume very long or first
        if d > 60: return 'Long'
        if d > 28: return 'Middle'
        return 'Short'
        
    df_sorted['int_type'] = df_sorted['interval_days'].apply(get_interval_type)
    
    # 3. Tataki Count (Run count since last Long interval)
    # Mark where interval > 60 as 1 (Reset), else 0 (Continue)
    # Or simply: if Long -> 1. if not -> prev + 1.
    
    # Identify "Start of Campaign" (Long interval or First run)
    df_sorted['is_campaign_start'] = df_sorted['interval_days'].apply(lambda x: 1 if x > 60 else 0)
    
    # Cumsum of campaign starts is group id? No.
    # We want count within campaign.
    # Group by (cumsum of is_campaign_start)
    df_sorted['campaign_id'] = df_sorted.groupby('horse_id')['is_campaign_start'].cumsum()
    
    # Cumcount within campaign
    # But we want 1-based index (1=Teppo)
    df_sorted['tataki_count'] = df_sorted.groupby(['horse_id', 'campaign_id']).cumcount() + 1
    
    # First run (interval=999) is also campaign start.
    
    # 4. Aptitude Stats (Expanding)
    # Group by [Horse, IntervalType]
    # CAUTION: We must shift(1) to avoid leakage.
    # Expanding Logic:
    #   For each row, calc mean of PREVIOUS rows with same Horse & IntType.
    
    group_cols = ['horse_id', 'int_type']
    
    df_sorted['apt_int_win'] = df_sorted.groupby(group_cols)['is_win'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(0)
    
    df_sorted['apt_int_top3'] = df_sorted.groupby(group_cols)['is_top3'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(0)
    
    df_sorted['apt_int_count'] = df_sorted.groupby(group_cols)['is_win'].transform(
        lambda x: x.expanding().count().shift(1)
    ).fillna(0)
    
    df_sorted['is_first_int_type'] = (df_sorted['apt_int_count'] == 0).astype(int)
    
    # 5. Return Features
    # Encode int_type?
    # Maybe OneHot or Ordinal? LightGBM handles category.
    # Let's map to int for simplicity.
    type_map = {'Short': 1, 'Middle': 2, 'Long': 3, 'New': 0}
    df_sorted['interval_type_code'] = df_sorted['int_type'].map(type_map)
    
    feats = [
        'interval_days',          # Raw days (useful for non-linear models)
        'interval_type_code',     # Categorical
        'tataki_count',           # Freshness
        'apt_int_win',            # WIN rate in this interval type
        'apt_int_top3',           # TOP3 rate in this interval type
        'is_first_int_type'       # Is first time
    ]
    keys = ['race_id', 'horse_number', 'horse_id']
    
    return df_sorted[keys + feats].copy()

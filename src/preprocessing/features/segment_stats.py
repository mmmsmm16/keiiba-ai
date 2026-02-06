
import pandas as pd
import numpy as np
import logging
from . import temporal_stats

logger = logging.getLogger(__name__)

def compute_segment_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block] M4-C: Weak Segment Stats
    - Small Field (<= 10 horses)
    - Mile (1400-1800m)
    - Interaction features
    """
    logger.info("ブロック計算中: segment_stats (M4-C)")
    
    req = ['horse_id', 'date', 'race_id', 'n_horses', 'distance', 'rank']
    if not all(c in df.columns for c in req):
        # n_horses might be missing if not merged yet? 
        # FeaturePipeline usually expects basic attributes.
        # If n_horses missing, try to count?
        if 'n_horses' not in df.columns:
             logger.warning("n_horses column missing. calculating from group count.")
             df['n_horses'] = df.groupby('race_id')['horse_id'].transform('count')
        
    df_sorted = df.sort_values(['horse_id', 'date']).copy()
    if not np.issubdtype(df_sorted['date'].dtype, np.datetime64):
        df_sorted['date'] = pd.to_datetime(df_sorted['date'])

    df_sorted['is_top3'] = (df_sorted['rank'] <= 3).astype(int)
    
    # helper
    keys = ['race_id', 'horse_number', 'horse_id']
    
    # 1. Small Field Stats (<= 10)
    # Filter
    # [Point] We want "Past performance IN small field" applied to ANY race.
    mask_small = df_sorted['n_horses'] <= 10
    df_small = df_sorted[mask_small].copy()
    
    # Rolling (Use generic helper if possible? or simple expanding)
    # Expanding is fine for "Career History in Small Field"
    # Or Rolling 365D if drift matters.
    # Let's use Expanding for stability (Small field samples are sparse).
    
    # Group by horse
    # shift(1) to avoid leakage
    grp_s = df_small.groupby('horse_id')
    df_small['small_n'] = grp_s['is_top3'].transform(lambda x: x.expanding().count().shift(1))
    df_small['small_top3_sum'] = grp_s['is_top3'].transform(lambda x: x.expanding().sum().shift(1))
    
    # Merge back to FULL df
    # Key: horse_id, date, race_id?
    # Need to be careful. A horse might have small field race, then large, then small.
    # The stats for the current race should be available regardless of current field size.
    # Wait.
    # If I am running in a Large Field today, I want to know my "Small Field Aptitude".
    # So I need to join "Latest Small Field Stats" to current record?
    # NO. I need "Stats calculated from PAST Small Field races".
    # This matches "as of date".
    # Since `df_small` only has rows for small field races, I can't just merge.
    # I need to propagate the last known state to all dates?
    # Alternatively:
    # 1. Create a Time Series of "Small Field Results" for each horse.
    # 2. For every race (Small or Large), lookup "Sum of Small Field Results before this date".
    
    # Optimized approach:
    # Create columns in df_sorted: 'is_small_field'
    df_sorted['is_small'] = (df_sorted['n_horses'] <= 10).astype(int)
    df_sorted['is_small_top3'] = (df_sorted['is_small'] & (df_sorted['is_top3']==1)).astype(int)
    
    # GroupBy Horse -> Expanding Sum of 'is_small' and 'is_small_top3'
    # shift(1)
    grp = df_sorted.groupby('horse_id')
    df_sorted['small_n_total'] = grp['is_small'].transform(lambda x: x.expanding().sum().shift(1)).fillna(0)
    df_sorted['small_top3_total'] = grp['is_small_top3'].transform(lambda x: x.expanding().sum().shift(1)).fillna(0)
    
    df_sorted['horse_small_top3_rate'] = (df_sorted['small_top3_total'] / df_sorted['small_n_total']).fillna(0)
    
    # 2. Mile Stats (1400 <= d <= 1800)
    mask_mile = (df_sorted['distance'] >= 1400) & (df_sorted['distance'] <= 1800)
    df_sorted['is_mile'] = mask_mile.astype(int)
    df_sorted['is_mile_top3'] = (df_sorted['is_mile'] & (df_sorted['is_top3']==1)).astype(int)
    
    df_sorted['mile_n_total'] = grp['is_mile'].transform(lambda x: x.expanding().sum().shift(1)).fillna(0)
    df_sorted['mile_top3_total'] = grp['is_mile_top3'].transform(lambda x: x.expanding().sum().shift(1)).fillna(0)
    
    df_sorted['horse_mile_top3_rate'] = (df_sorted['mile_top3_total'] / df_sorted['mile_n_total']).fillna(0)
    
    # 3. Interaction (FieldSize * PacePressure)
    # Pace Pressure must be present. If not, skip or use simplified.
    # Assuming 'avg_pace_pressure' or similar is in pipeline... 
    # Actually, this block shouldn't depend on other computed blocks if possible (order dependency).
    # If Pace Stats is separate, we might not have it here yet unless we load it.
    # FeaturePipeline loads blocks. If we need interaction of two blocks, maybe do it in a separate "Interaction Block" or post-process.
    # Or, if this function receives `df` which is `base_df`, it doesn't have other features.
    # So we skip Pace interaction here if it requires 'pace_pressure' feature.
    # Unless we compute it on the fly? Too heavy.
    # Let's stick to simple interactions available from raw data or simple stats.
    # Field Size is in raw.
    # If we want "Small Field x Jockey Win Rate", we can do it.
    # For now, just the Rates.
    
    final_feats = ['horse_small_top3_rate', 'horse_mile_top3_rate', 'small_n_total', 'mile_n_total']
    
    return df_sorted[keys + final_feats].copy()

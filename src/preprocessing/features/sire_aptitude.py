
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_sire_aptitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block] Sire Aptitude (血統適性)
    - 種牡馬(sire_id)および母父(damsire_id - if available)のコース別・距離別成績を集計する。
    - Target Encodingの一種であるため、Leakage防止（Expanding Mean + Shift）が必須。
    
    Features:
    - sire_course_win_rate: 種牡馬の当該コース(course_id)における累積勝率。
    - sire_dist_win_rate: 種牡馬の当該距離カテゴリ(distance_category)における累積勝率。
    - sire_type_win_rate: 種牡馬の当該芝ダ区分(surface)における累積勝率。
    - (Option) damsire versions if data available. (Assuming we might not have damsire_id easily linked here, 
      but if 'bloodline_stats' block exists, maybe we can access it? 
      Actually, standard JraVanDataLoader provides 'sire_id'. 'damsire_id' might be in bloodline info.
      Let's stick to 'sire_id' which is definitely in columns.)
    """
    logger.info("ブロック計算中: compute_sire_aptitude")
    
    keys = ['race_id', 'horse_number', 'horse_id', 'date']
    df_sorted = df.sort_values(['date', 'race_id', 'horse_number']).copy()

    # Required cols check
    req_for_core = ['sire_id', 'venue', 'distance', 'surface']
    for c in req_for_core:
        if c not in df_sorted.columns:
            logger.warning(f"compute_sire_aptitude: missing {c}, returning keys only.")
            return df_sorted[[k for k in keys if k in df_sorted.columns]].copy()

    # Context keys
    df_sorted['course_id'] = (
        df_sorted['venue'].astype(str) + '_' +
        df_sorted['surface'].astype(str) + '_' +
        df_sorted['distance'].astype(str)
    )
    df_sorted['dist_bin'] = (pd.to_numeric(df_sorted['distance'], errors='coerce').fillna(0) // 100) * 100

    # Targets (safe for mixed train/inference rows)
    rank_num = pd.to_numeric(df_sorted['rank'], errors='coerce') if 'rank' in df_sorted.columns else pd.Series(np.nan, index=df_sorted.index)
    df_sorted['is_valid_result'] = rank_num.notna().astype(int)
    df_sorted['is_win'] = (rank_num == 1).astype(float)
    df_sorted['is_top3'] = (rank_num <= 3).astype(float)
    df_sorted['is_win_filled'] = df_sorted['is_win'].fillna(0.0)
    df_sorted['is_top3_filled'] = df_sorted['is_top3'].fillna(0.0)

    prior = 10.0
    default_global = 0.08

    # Leakage-safe global prior
    global_wins = df_sorted['is_win_filled'].cumsum().shift(1).fillna(0.0)
    global_counts = df_sorted['is_valid_result'].cumsum().shift(1).fillna(0.0)
    df_sorted['global_win_prior'] = np.where(global_counts > 0, global_wins / global_counts, default_global)

    def past_rate(group_cols):
        grp = df_sorted.groupby(group_cols)
        count = grp['is_valid_result'].transform(lambda x: x.cumsum().shift(1)).fillna(0.0)
        win_sum = grp['is_win_filled'].transform(lambda x: x.cumsum().shift(1)).fillna(0.0)
        raw = np.where(count > 0, win_sum / count, np.nan)
        smooth = (win_sum + prior * df_sorted['global_win_prior']) / (count + prior)
        return count, raw, smooth

    # 1) Core rates
    cnt_course, raw_course, sm_course = past_rate(['sire_id', 'course_id'])
    cnt_dist, raw_dist, sm_dist = past_rate(['sire_id', 'dist_bin'])
    cnt_surface, raw_surface, sm_surface = past_rate(['sire_id', 'surface'])
    df_sorted['sire_course_win_rate'] = sm_course
    df_sorted['sire_dist_win_rate'] = sm_dist
    df_sorted['sire_surface_win_rate'] = sm_surface

    # 2) Time-decayed versions (EWM over past starts)
    alpha = 0.18
    df_sorted['sire_course_win_rate_decay'] = df_sorted.groupby(['sire_id', 'course_id'])['is_win_filled'].transform(
        lambda x: x.shift(1).ewm(alpha=alpha, adjust=False, min_periods=1).mean()
    )
    df_sorted['sire_dist_win_rate_decay'] = df_sorted.groupby(['sire_id', 'dist_bin'])['is_win_filled'].transform(
        lambda x: x.shift(1).ewm(alpha=alpha, adjust=False, min_periods=1).mean()
    )
    df_sorted['sire_surface_win_rate_decay'] = df_sorted.groupby(['sire_id', 'surface'])['is_win_filled'].transform(
        lambda x: x.shift(1).ewm(alpha=alpha, adjust=False, min_periods=1).mean()
    )

    # 3) Reproducibility / stability (dispersion)
    df_sorted['sire_win_rate_std_50'] = df_sorted.groupby('sire_id')['is_win_filled'].transform(
        lambda x: x.shift(1).rolling(50, min_periods=5).std()
    )
    q75 = df_sorted.groupby('sire_id')['is_win_filled'].transform(
        lambda x: x.shift(1).rolling(50, min_periods=8).quantile(0.75)
    )
    q25 = df_sorted.groupby('sire_id')['is_win_filled'].transform(
        lambda x: x.shift(1).rolling(50, min_periods=8).quantile(0.25)
    )
    df_sorted['sire_win_rate_iqr_50'] = q75 - q25

    # 4) Bloodline x track variant fit
    if 'track_variant' in df_sorted.columns:
        tv = pd.to_numeric(df_sorted['track_variant'], errors='coerce').fillna(0.0)
        df_sorted['track_variant_num'] = tv
        grp_sire = df_sorted.groupby('sire_id')
        df_sorted['variant_weighted'] = df_sorted['track_variant_num'] * df_sorted['is_top3_filled']
        pref_num = grp_sire['variant_weighted'].transform(lambda x: x.cumsum().shift(1)).fillna(0.0)
        pref_den = grp_sire['is_top3_filled'].transform(lambda x: x.cumsum().shift(1)).fillna(0.0)
        df_sorted['sire_variant_pref'] = np.where(pref_den > 0, pref_num / pref_den, 0.0)
        tau = 0.8
        df_sorted['blood_variant_fit'] = np.exp(-np.abs(df_sorted['track_variant_num'] - df_sorted['sire_variant_pref']) / tau)
    else:
        df_sorted['sire_variant_pref'] = 0.0
        df_sorted['blood_variant_fit'] = 0.5

    cols = [
        'sire_course_win_rate',
        'sire_dist_win_rate',
        'sire_surface_win_rate',
        'sire_course_win_rate_decay',
        'sire_dist_win_rate_decay',
        'sire_surface_win_rate_decay',
        'sire_win_rate_std_50',
        'sire_win_rate_iqr_50',
        'sire_variant_pref',
        'blood_variant_fit'
    ]
    df_sorted[cols] = df_sorted[cols].fillna(default_global)
    return df_sorted[keys + cols].copy()

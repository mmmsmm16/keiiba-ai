
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_rolling_stats(
    df: pd.DataFrame,
    group_col: str,
    target_cols: list,
    time_col: str = 'date',
    windows: list = ['180D', '365D'],
    min_periods: int = 1
) -> pd.DataFrame:
    """
    時系列データの移動集計 (Rolling Stats) を計算する最適化関数。
    - 厳密なソート (group_col, time_col)
    - 日次集計 -> Rolling -> Merge (高速化とIndex整合性)
    - closed='left' (当日を含まない)
    """
    # 必要なカラムのチェック
    req_cols = [group_col, time_col] + target_cols
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for rolling stats: {missing}")

    # 作業用DF作成 (ソート済み)
    work_df = df[req_cols].copy()
    if not np.issubdtype(work_df[time_col].dtype, np.datetime64):
        work_df[time_col] = pd.to_datetime(work_df[time_col])
    
    # 日次集計 (同一日のレースをまとめる)
    # これにより groupby().rolling() のインデックス重複問題を回避
    # count用ダミー
    work_df['__cnt'] = 1
    
    agg_funcs = {col: 'sum' for col in target_cols}
    agg_funcs['__cnt'] = 'sum' # Daily Race Count
    
    # Group by [Group, Date] -> Daily Summary
    daily_agg = work_df.groupby([group_col, time_col]).agg(agg_funcs)
    # daily_agg index: MultiIndex (group, date)
    
    # Rolling計算用DataFrame (Index: group, date)
    daily_stats = pd.DataFrame(index=daily_agg.index)
    
    # Daily Aggregationに対してRollingを適用
    # level=0 (group) でグルーピングし、日付(level=1)に基づいてRolling
    # on引数は使わず、indexがdatetimeであることを利用
    
    # Note: groupby().rolling() on hierarchical index works if level 1 is datetime?
    # No, usually need to reset index.
    
    # Resetting index is safest
    temp_df = daily_agg.reset_index(level=0) # Index=Date, Col=Group
    
    # This creates a reusable roller? No, window is specific.
    
    for window in windows:
        logger.info(f"  Computing rolling {window} stats for {group_col}...")
        r = temp_df.groupby(group_col).rolling(window, closed='left', min_periods=min_periods)
        
        prefix = window.lower()
        
        # 1. Race Count
        # __cnt の sum が「期間内の総レース数」
        s_count = r['__cnt'].sum()
        # s_count index: (group, date). Matches daily_agg index.
        daily_stats[f"{group_col}_n_races_{prefix}"] = s_count
        
        # 2. Target Sums
        for tgt in target_cols:
            s_sum = r[tgt].sum()
            daily_stats[f"{group_col}_{tgt}_sum_{prefix}"] = s_sum
            
    # Merge back to original DataFrame
    # df と daily_stats を (group_col, time_col) で結合
    # df側は同一キーで複数行あるが、daily_stats側はユニークなので left join で増幅されることなく値が付与される
    
    # daily_stats is MultiIndex (group, date). 
    # Left join on index?
    # df: reset_index? No.
    
    # Simply join on columns.
    daily_stats = daily_stats.reset_index() # columns: group, date, stats...
    
    # Ensure types match for merge
    # df[group_col] might be numeric or string. daily_stats[group_col] should be same.
    
    # Optimize: set index on df for join?
    # Merge is standard.
    
    merged_df = pd.merge(df, daily_stats, on=[group_col, time_col], how='left')
    
    # Return ONLY the new columns (plus key for sanity check if needed, but usually we return full or partial)
    # Caller expects keys + feats.
    
    new_cols = [c for c in daily_stats.columns if c not in [group_col, time_col]]
    
    # FillNa with 0 (no history)
    merged_df[new_cols] = merged_df[new_cols].fillna(0)
    
    return merged_df


def compute_relative_stats(
    df: pd.DataFrame,
    target_cols: list,
    time_col: str = 'date',
    window: int = 10000,
    group_keys: list = None,
    use_fixed_baseline: bool = True,
    baseline_cutoff_date: str = '2024-12-31'
) -> pd.DataFrame:
    """
    (M4-A) 時系列相対化指標。
    当該レース時点での「直近N件の全体平均・標準偏差」を算出し、Z-scoreを計算する。
    これにより、インフレする変数（n_races等）の「その時点での相対位置」を数値化する。
    
    Args:
        window: Sample count for rolling window (not days). 
                e.g., 10000 samples (records) approx 1-2 months of data?
                JRA has ~50k rows/year. 10k is ~2.5 months.
        use_fixed_baseline: If True, use fixed mean/std from data before baseline_cutoff_date.
                           This prevents drift when applying to new data.
        baseline_cutoff_date: Date to use for computing fixed baseline (typically end of training period).
    """
    # ソート
    df_sorted = df.sort_values(time_col).copy()
    
    # 計算結果格納用
    res_df = pd.DataFrame(index=df.index)
    
    # Ensure datetime
    if not np.issubdtype(df_sorted[time_col].dtype, np.datetime64):
        df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])
    
    # Fixed Baseline Mode: Compute mean/std from historical data only
    if use_fixed_baseline:
        cutoff = pd.to_datetime(baseline_cutoff_date)
        baseline_df = df_sorted[df_sorted[time_col] <= cutoff]
        
        if len(baseline_df) < 1000:
            logger.warning(f"Fixed baseline has only {len(baseline_df)} samples. Using rolling instead.")
            use_fixed_baseline = False
        else:
            # Compute fixed baseline statistics
            fixed_means = baseline_df[target_cols].mean()
            fixed_stds = baseline_df[target_cols].std().replace(0, 1.0)
            logger.info(f"Using fixed baseline from data <= {baseline_cutoff_date} ({len(baseline_df)} samples)")
            
            # Apply z-score with fixed baseline
            z_scores = (df_sorted[target_cols] - fixed_means) / fixed_stds
            z_scores.columns = [f"{c}_relative_z" for c in target_cols]
            z_scores = z_scores.fillna(0)
            
            # Assign back using original index
            for col in z_scores.columns:
                res_df.loc[df_sorted.index, col] = z_scores[col].values
            
            return res_df
    
    # Rolling Mode (original behavior)
    # groupbyが必要な場合 (例: Grade別、Venue別など)
    # 今回は「全体」に対する相対化が主目的だが、group_keysがあれば対応
    
    iterator = [(None, df_sorted)]
    if group_keys:
        iterator = df_sorted.groupby(group_keys)
        
    for name, group in iterator:
        # Rolling stats
        # min_periods needs to be large enough to stabilize std
        min_p = min(window // 10, 100)
        
        # closed='left' is not supported for integer window (count based) in older pandas?
        # It is supported.
        # But 'step' based rolling is just .rolling(window).
        # To avoid self-leak perfectly, we ideally use closed='left'.
        # But standard rolling implies current is included.
        # For 'Global stats' of 10000 samples, including self is tiny.
        # But let's try shift(1) to be strictly 'past'.
        
        roller = group[target_cols].shift(1).rolling(window=window, min_periods=min_p)
        
        means = roller.mean()
        stds = roller.std().replace(0, 1.0) # Avoid div0
        
        z_scores = (group[target_cols] - means) / stds
        
        # Rename: e.g. "jockey_n_races" -> "jockey_n_races_relative_z"
        z_scores.columns = [f"{c}_relative_z" for c in target_cols]
        
        # FillNa with 0 (average)
        z_scores = z_scores.fillna(0)
        
        # Assign to result (matching index)
        # Note: res_df is aligned with df, group is subset.
        res_df.loc[group.index, z_scores.columns] = z_scores
        
    return res_df



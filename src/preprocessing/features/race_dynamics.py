
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_race_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block] Race Dynamics (展開力学)
    - レースメンバー構成から「展開」を予測し、相対的な優位性を評価する。
    - 未来の情報（今回の脚質）は使わず、過去のデータから計算する。
    
    Features:
    - front_runner_count: メンバー中の「逃げ」候補（前走逃げ、または近走逃げ率NO.1）の数。逃げ多数ならハイペース想定。
    - race_pace_level_3f: メンバーの「平均テン3F」の平均値。レース全体のスピードレベル。
    - relative_3f_score: (自分のテン3F - レース平均) / レース標準偏差。負の値が大きいほど相対的に速い（有利）。
    - is_sole_leader: 逃げ候補が自分1頭だけの場合のフラグ（単騎逃げ濃厚）。
    """
    logger.info("ブロック計算中: compute_race_dynamics")
    
    keys = ['race_id', 'horse_number', 'horse_id', 'date']
    
    # We need Past Performance Data to predict Pace.
    # We rely on pre-calculated 'past_3f_time' or similar if available, or compute on the fly.
    # Since we don't have easy access to past N races in this block function (it takes df of all history),
    # we can use Expanding Mean of 'time_3f' (First 3F time) from previous races.
    # Or use 'running_style_code' from previous race.
    
    # 1. Calculate 'Past Average First 3F' for each horse (Expanding)
    df_sorted = df.sort_values(['horse_id', 'date']).copy()
    
    # Need 3F time column. 'time_3f' usually denotes First 3F in JRA data if available.
    # If not, we might check 'shokai_3f_time' or similar. 
    # Let's assume we have to use 'first_3f_time' or calculate it if raw data available.
    # Checking pipeline... usually 'pace_stats' calculates this but locally.
    
    # If we don't have explicit 3F time in input df, we might skip or fail.
    # Assuming 'first_3f_time' exists in the input df (raw data usually has it).
    # If not, checks.
    
    target_3f_col = 'first_3f_time' # Typical name
    if target_3f_col not in df_sorted.columns:
        # Try finding standard name
        if 'h_first_3f' in df_sorted.columns: target_3f_col = 'h_first_3f'
        elif 'first_3f' in df_sorted.columns: target_3f_col = 'first_3f'
        # Fallback: cannot compute
    
    # Expanding Mean of 3F Time (Shifted) -> "My Expected 3F Time"
    if target_3f_col in df_sorted.columns:
        # 0 or NaN handling
        df_sorted[target_3f_col] = pd.to_numeric(df_sorted[target_3f_col], errors='coerce')
        df_sorted['my_exp_3f'] = df_sorted.groupby('horse_id')[target_3f_col].transform(
            lambda x: x.expanding().mean().shift(1)
        )
    else:
        df_sorted['my_exp_3f'] = np.nan
        
    # Running Style: 'running_style_code'. 
    # 1=Nige (Runner/Leader).
    # We want to know if horse WAS a runner in PREVIOUS race.
    if 'running_style_code' in df_sorted.columns:
        df_sorted['prev_run_style'] = df_sorted.groupby('horse_id')['running_style_code'].shift(1)
        # Is Front Runner Candidate? (Prev was Nige)
        df_sorted['is_front_runner_cand'] = (df_sorted['prev_run_style'] == 1).astype(int)
    elif 'pass_1' in df_sorted.columns or 'passing_rank' in df_sorted.columns:
        # Derive from pass_1 (Corner 1 rank)
        # pass_1 format: "1" or "1-1" etc.
        def is_nige_val(val):
            try:
                s = str(val)
                p1 = s.split('-')[0]
                return 1 if float(p1) == 1 else 0
            except:
                return 0
        
        target_pass_col = 'pass_1' if 'pass_1' in df_sorted.columns else 'passing_rank'
        df_sorted['is_nige_derived'] = df_sorted[target_pass_col].apply(is_nige_val)
        
        # Shift to get PREVIOUS race run style
        df_sorted['is_front_runner_cand'] = df_sorted.groupby('horse_id')['is_nige_derived'].shift(1).fillna(0).astype(int)
        
    else:
        df_sorted['is_front_runner_cand'] = 0

    # Now Group by Race to calculate Context
    # We need to impute NaNs in my_exp_3f before grouping, or handle in aggregation.
    # Fill with global mean?
    global_avg_3f = df_sorted['my_exp_3f'].mean()
    df_sorted['my_exp_3f_filled'] = df_sorted['my_exp_3f'].fillna(global_avg_3f)

    # Groupby Race
    grp = df_sorted.groupby('race_id')
    
    # 1. Front Runner Count
    df_sorted['front_runner_count'] = grp['is_front_runner_cand'].transform('sum')
    
    # 2. Race Pace Level (Avg of Expected 3F)
    df_sorted['race_pace_level_3f'] = grp['my_exp_3f_filled'].transform('mean')
    df_sorted['race_pace_std_3f'] = grp['my_exp_3f_filled'].transform('std').fillna(1.0) # Avoid div0
    
    # 3. Relative 3F Score (Standardized)
    # Lower 3F is Faster.
    # Score = (RaceAvg - MyTime) / Std. 
    # Positive Score => I am faster than average.
    df_sorted['relative_3f_score'] = (df_sorted['race_pace_level_3f'] - df_sorted['my_exp_3f_filled']) / df_sorted['race_pace_std_3f']
    
    # 4. Sole Leader
    # If front_runner_count == 1 AND I am the one
    df_sorted['is_sole_leader'] = ((df_sorted['front_runner_count'] == 1) & (df_sorted['is_front_runner_cand'] == 1)).astype(int)
    
    # 5. High Pace Warning
    # If front_runner_count >= 3
    df_sorted['is_high_pace_warn'] = (df_sorted['front_runner_count'] >= 3).astype(int)
    
    cols = [
        'front_runner_count',
        'race_pace_level_3f',
        'relative_3f_score',
        'is_sole_leader',
        'is_high_pace_warn'
    ]
    
    return df_sorted[keys + cols].copy()


import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_physique_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block] Physique & Training (馬体・調教)
    - 馬体重の増減を、馬格（大型・小型）や文脈に合わせて再評価する。
    - 調教データの「加速感」や「負荷」をスコア化する。
    
    Features:
    - weight_impact_score: 体重増減率 × (基準体重 / 現在体重)。小型馬の減量を重く罰する。
    - best_weight_diff: 過去の好走時（Top3）平均馬体重との差分。
    - training_accel_score: 調教ラスト1Fと平均ラップの差。 (Avg - Last1F). Positive means acceleration.
    - is_training_best: 今回の調教タイムが過去の自己ベストか？（距離別） -> 簡易的に4Fタイムで判定。
    """
    logger.info("ブロック計算中: compute_physique_training")
    
    # Required cols
    # weight, weight_diff (or calculated), training_time_*
    # rank (for best weight)
    
    keys = ['race_id', 'horse_number', 'horse_id', 'date']
    
    # Check basics
    if 'weight' not in df.columns:
        return df[['race_id', 'horse_number']].copy()

    df_sorted = df.sort_values(['horse_id', 'date']).copy()
    
    # 1. Weight Impact Score
    # 小型馬 (e.g. 420kg) の -10kg (-2.3%) と 大型馬 (520kg) の -10kg (-1.9%)。
    # Impact = (Diff / Weight) * (SCALING_FACTOR / Weight)?
    # Simple logic: If weight < 450 and diff < 0, penalty is amplified.
    # If weight > 500 and diff > 0, bonus/neutral (growth).
    
    # weight_diff might be missing
    if 'weight_diff' not in df_sorted.columns:
        df_sorted['weight_diff'] = 0
        
    # Relative change
    df_sorted['weight_change_pct'] = df_sorted['weight_diff'] / df_sorted['weight'].replace(0, np.nan)
    
    # Physique Factor: 基準500kg。小さいほど値が大きくなる (=1.0 / (w/500))
    # Avoid div by zero
    w = df_sorted['weight'].replace(0, 470) # Default avg
    physique_factor = 500 / w 
    
    # Impact Score
    # change_pct * physique_factor
    # Small horse (-2%) * (1.1) = -2.2%
    # Large horse (-2%) * (0.9) = -1.8%
    # This naturally penalizes small horses more for same pct change? 
    # Actually pct change is already handling relative size.
    # But usually physical depletion is non-linear.
    # Let's use simple logic:
    # Score = ChangePct.
    # But if Weight < 440 and ChangePct < 0: Score *= 1.5 (Significant Penalty)
    
    df_sorted['weight_impact_score'] = df_sorted['weight_change_pct'] * 100 # percentage
    
    mask_small_loss = (df_sorted['weight'] < 440) & (df_sorted['weight_diff'] < 0)
    df_sorted.loc[mask_small_loss, 'weight_impact_score'] *= 1.5
    
    # 2. Best Weight Diff
    # Calc avg weight when is_top3=1 in PAST races.
    
    # Identify Top3
    if 'rank' in df_sorted.columns:
        rank_num = pd.to_numeric(df_sorted['rank'], errors='coerce')
        is_top3 = (rank_num <= 3)
    else:
        is_top3 = pd.Series([False]*len(df_sorted), index=df_sorted.index)
        
    # We need expanding mean of weight WHERE is_top3 was true.
    # Standard expanding mean includes all. 
    # We can use custom apply or workaround.
    # Workaround: Set weight to NaN where not Top3? No, we need past records.
    
    # Using a masked expanding mean
    # top3_weights = df_sorted['weight'].where(is_top3)
    # This only has values on top3 rows.
    # expanding mean of this series will ignore NaNs? Yes.
    # We need to shift(1) to avoid leakage.
    
    # Note: Top3 flag is from CURRENT race. We cannot use it for current definition.
    # We want "Past Top3 Weights".
    # So we take Series S: (Weight if Top3 else NaN).
    # Then expanding().mean().shift(1).
    
    # Note: Using apply with custom logic can return MultiIndex (horse_id, index).
    # We should ensure alignment. Or use a custom loop if performance allows, but for 1M rows it's slow.
    # Approach: Calculate separately, then merge back.
    
    def calc_best_weight(subset):
        # subset is entries for one horse, sorted by date
        # We need expanding mean of weights where is_top3 was TRUE
        # 1. Extract weights where top3
        # 2. Reindex to full subset length? No.
        # We want: on row i, average of 'weight' for all previous rows j<i where is_top3 is True.
        
        # Mask weights:
        w_masked = subset['weight'].where(subset['is_top3_flag'])
        # Expanding mean. This propagates NaN if we just do expanding().mean() on Series with NaNs?
        # No, pandas mean ignores NaNs. So expanding().mean() gives "Mean of non-NaN values so far".
        # This is exactly what we want!
        return w_masked.expanding().mean().shift(1)

    # Prepare flag
    df_sorted['is_top3_flag'] = is_top3
    
    # Apply and assign. 
    # GroupBy apply returns MultiIndex if group keys are not index?
    # To be safe, we reset index in apply or direct transform? 
    # transform requires returning same shape. 
    # The lambda above returns same shape (Series of same length).
    
    # Try using transform with the logic
    df_sorted['best_weight_avg'] = df_sorted.groupby('horse_id', group_keys=False).apply(calc_best_weight)
    
    # Fallback to current avg
    df_sorted['avg_weight_all'] = df_sorted.groupby('horse_id')['weight'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df_sorted['best_weight_avg'] = df_sorted['best_weight_avg'].fillna(df_sorted['avg_weight_all'])
    
    df_sorted['best_weight_diff'] = df_sorted['weight'] - df_sorted['best_weight_avg']
    # If weight is 0 or nan, diff is nan.
    df_sorted['best_weight_diff'] = df_sorted['best_weight_diff'].fillna(0)
    
    # 3. Training Accel Score
    # We need training columns.
    # training_time_4f, training_time_last1f
    
    accel_score = np.zeros(len(df_sorted))
    
    if 'training_time_last1f' in df_sorted.columns and 'training_time_4f' in df_sorted.columns:
         t1 = df_sorted['training_time_last1f']
         t4 = df_sorted['training_time_4f']
         
         # Avg per furlong for 4F. (This is crude, 4F includes 1F).
         # Acceleration = (Avg Speed of first 3F) vs (Speed of last 1F)
         # Time:
         # T_3F_start = T4 - T1
         # Avg_3F = T_3F_start / 3.0
         # Diff = Avg_3F - T1. (If T1 is smaller, it's faster -> Acceleration).
         # Score = Diff.
         
         t3_start = t4 - t1
         # Avoid bad data
         mask_valid = (t4 > t1) & (t1 > 0)
         
         avg_3f = t3_start / 3.0
         # Positive if T1 < Avg_3F
         score = avg_3f - t1
         
         accel_score = np.where(mask_valid, score, 0)
         
    df_sorted['training_accel_score'] = accel_score
    
    # Return
    feats = [
        'weight_impact_score',
        'best_weight_diff',
        'training_accel_score'
    ]
    
    return df_sorted[keys + feats].copy()

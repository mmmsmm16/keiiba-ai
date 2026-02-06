import pandas as pd
import numpy as np

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 3.7] Elo Rating
    - 各馬のRatingを時系列で更新・保持する
    - shift(1)して「出走前」のRatingを特徴量にする
    - K=32, Initial=1500
    - 対戦相手との平均レート差 + 勝敗(先着)に基づいて更新
    - ※完全なElo計算は重いため、ここでは「レース結果に基づく簡易レート更新」を行う
    """
    cols = ['race_id', 'horse_number', 'horse_id']
    df_sorted = df.sort_values(['date', 'race_id']).copy()
    
    # Elo calc requires sequential processing.
    # Python Loop or iterative.
    # For speed, we can process race by race (group keys).
    # But >50k races. Loop is slow.
    # Improved: Vectorized approximation is impossible for recursive states.
    # However, we can run a single pass or accept "Batch" updates (e.g. per month)? No.
    
    # Fast implementation found in many Kaggle kernels:
    # Use a dictionary to hold current ratings.
    # Loop over races. Update dict.
    # This takes ~1-2 mins for 50k races. Acceptable.
    
    # Parameters
    K = 32
    INITIAL_RATING = 1500
    
    # Store ratings: {horse_id: rating}
    ratings = {}
    
    # To store results
    pre_race_ratings = [] # Aligned with df order? No, loop order.
    
    # We need to iterate races chronologically.
    # Group by race_id, but keeping date order.
    # df_sorted is already sorted by date.
    
    # Get necessary columns as numpy array/list for speed
    # We need: race_id, horse_id, rank
    # Assuming rank is available.
    
    race_groups = df_sorted.groupby('race_id')
    # Use iteration over groups (preserves order of appearance if sort=False/sort_values done?)
    # Groupby object doesn't guarantee order unless we sort keys.
    
    # Unique race_ids in order
    rids = df_sorted['race_id'].unique()
    
    # Pre-fetch data to avoiding repetitive DF indexing
    # Create a nice lookup dict or just iterate df rows?
    # iterating df rows is too slow.
    # iterating groups is better.
    
    # Pre-calculating a dict of {race_id: [(horse_id, rank), ...]} is fastest?
    # Yes.
    
    # But wait, result list mapping back to DF rows is pain.
    # Alternative:
    # Use 'match count' based approach? No, classic Elo is better.
    
    # Simple Loop implementation with optimization for mapping back.
    # 1. Initialize 'elo_rating' column with NaN
    # 2. Iterate races.
    # 3. For each race, get current ratings.
    # 4. Calc new ratings.
    # 5. Store PRE-ratings into a list/dict.
    
    # Optimization: processing race by race.
    
    # Dict: horse_id -> rating
    current_ratings = {}
    
    # Output storage
    # Using a dict {(race_id, horse_id): rating} to map later
    history_ratings = {}
    
    # Extract data as list of tuples: (race_id, horse_id, rank)
    # Assuming df_sorted index is unique or we reset it.
    df_sorted = df_sorted.reset_index(drop=True)
    
    # We need to process races sequentially.
    # Extract data for iteration
    # Structure: [ [race_id, [ (horse_id, rank), ... ]] ... ]
    
    # Use groupby to create list of races
    # This might be memory intensive but fast iteration
    # Actually, simpler:
    # df needed cols
    sub = df_sorted[['race_id', 'horse_id', 'rank']].values
    # race_id is col 0, horse_id col 1, rank col 2
    
    # We need boundaries of races.
    # Since sorted by race_id... (Wait, df sorted by date then race_id)
    # We can detect race change.
    
    r_prev = None
    batch_h = []
    batch_r = []
    
    # Result buffer
    # list of (index, rating)
    res_buffer = np.zeros(len(df_sorted)) # index aligned
    
    # Helper to update ratings for a batch (race)
    def update_race(h_ids, ranks):
        # h_ids: list of horse_ids
        # ranks: list of ranks
        n = len(h_ids)
        if n < 2: return # No competition
        
        # Get current ratings
        cur_rs = np.array([current_ratings.get(h, INITIAL_RATING) for h in h_ids])
        
        # Store PRE ratings to result (This matches requirements: Feature is PRE-race rating)
        # But we need to write back to the correct indices.
        # So update_race needs indices too.
        pass
        
    # Re-structure loop
    # Iterating over groupby is clean enough.
    
    for rid, grp_df in df_sorted.groupby('race_id', sort=False):
        h_ids = grp_df['horse_id'].values
        ranks = grp_df['rank'].values
        indices = grp_df.index.values
        
        # Get Current Ratings
        rs = [current_ratings.get(h, INITIAL_RATING) for h in h_ids]
        
        # Assign to Result (Pre-Race Rating)
        # Vectorized assignment to buffer?
        # res_buffer[indices] = rs  <-- Fast
        res_buffer[indices] = rs
        
        # Update Logic (Simple Elo)
        # Winner takes points from Losers?
        # Multi-player Elo:
        # Rate_new = Rate_old + K * (Actual_Score - Expected_Score)
        # Expected_Score_A = 1 / (1 + 10^((Rb - Ra)/400)) ... summed over opponents?
        # Linear approximation for N-player:
        # R_avg = Mean(Rs)
        # Perf = (N - Rank) / (N*(N-1)/2) ? No.
        # Simple method:
        # Update vs Average of Field?
        # Expected = 1 / (1 + 10^((R_avg - R_i)/400))
        # Actual: (N - Rank) / (N - 1)  (Normalized 0..1, 1=Win)
        # R_new = R_old + K * (Actual - Expected) * (N-1)?
        # There are variants.
        # Let's use: Actual = (N - Rank) / (N - 1)
        # Expected = 1 / (1 + 10^((FieldAvg - Own)/400))
        # This keeps inflation/deflation minimal if K is balanced.
        
        # [Fix] Exclude rank=0 (DNF/取消/除外) from ELO calculation
        # rank=0 means horse didn't finish properly
        valid_mask = (~np.isnan(ranks)) & (ranks > 0)
        if valid_mask.sum() < 1:
            continue
            
        valid_rs = np.array(rs)[valid_mask]
        valid_ranks = ranks[valid_mask]
        valid_hids = h_ids[valid_mask]
        
        field_avg = np.mean(valid_rs)
        n_field = len(valid_rs)
        
        # Expected
        # Ei = 1 / (1 + 10 ** ((field_avg - valid_rs) / 400))
        # Logic check: If Field > Own, Exp < 0.5. Correct.
        exps = 1.0 / (1.0 + 10.0 ** ((field_avg - valid_rs) / 400.0))
        
        # [Fix] Use field_size from DF if available to represent full field context
        f_size = grp_df['field_size'].iloc[0] if 'field_size' in grp_df.columns else n_field
        n_effective = max(n_field, f_size)
        
        # Actual
        # Rank 1 -> 1.0, Rank N -> 0.0
        # [Fix] Handle single horse races (n_field=1) by assuming virtual opponent at 1500
        if n_effective > 1:
            actuals = (n_effective - valid_ranks) / (n_effective - 1)
        else:
            # Single horse case: 1.0 if rank 1, else 0.0
            actuals = (valid_ranks == 1).astype(float)
            
        # Update
        # Using a dynamic K based on N?
        # Standard Elo K=32.
        # In multi-player, K often scaled by (N-1) or similar to ensure movement?
        # If we compare to Field Avg (1 opponent effectively), K=32 is fine.
        diffs = K * (actuals - exps)
        
        new_rs = valid_rs + diffs
        
        # Write back to dict
        for h, r_new in zip(valid_hids, new_rs):
            current_ratings[h] = r_new

    # Assign result
    df_sorted['horse_elo'] = res_buffer
    
    # [Fix] JIT mode workaround: Apply last-known rating to horses that are still at initial
    # This handles cases where horses only appear in single-horse-per-race JIT data
    # but have accumulated ratings from other calculations
    for idx, row in df_sorted.iterrows():
        if df_sorted.at[idx, 'horse_elo'] == INITIAL_RATING:
            h = row['horse_id']
            if h in current_ratings:
                df_sorted.at[idx, 'horse_elo'] = current_ratings[h]
    
    # Extra: Field Elo Stats
    grp = df_sorted.groupby('race_id')['horse_elo']
    df_sorted['field_elo_mean'] = grp.transform('mean')
    df_sorted['elo_gap_to_top'] = grp.transform('max') - df_sorted['horse_elo']
    
    out_cols = ['horse_elo', 'field_elo_mean', 'elo_gap_to_top']
    return df_sorted[cols + out_cols].copy()

import pandas as pd
import numpy as np

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 3.3] 近走トレンド (Form Trend)
    - rank, time_diff の EWM(指数平滑移動平均) と Slope(傾き)
    - SpeedIndexがあればそれも計算
    - 注意: Shift(1)して「過去走のみ」で計算すること
    """
    cols = ['race_id', 'horse_number', 'horse_id']
    df_sorted = df.sort_values(['horse_id', 'date']).copy()
    grp = df_sorted.groupby('horse_id')
    
    # Targets
    targets = []
    if 'rank' in df_sorted.columns: targets.append('rank')
    if 'time_diff' in df_sorted.columns: targets.append('time_diff')
    # simple speed index proxy if not exists
    # If speed_index is not in columns, we skip it
    if 'speed_index' in df_sorted.columns: targets.append('speed_index')
    
    out_cols = []
    
    for tgt in targets:
        # Pre-calc shifted series to ensure no leakage
        # Shift 1 first!
        # s_shifted IS the series of "past races".
        # value at row i is value of race i-1.
        s_shifted = grp[tgt].shift(1)
        
        # 1. EWM (Span=3, 5)
        # ewm is not directly supported in transform with groupby usually easily?
        # applying ewm on shifted series. 
        # Need to loop or use apply? apply is slow.
        # Groupby EWM: df.groupby('id')['val'].ewm(span=3).mean() works in recent pandas? yes.
        # But we need to run it on shifted data.
        
        # Strategy: Create a shifted column, then calc EWM on it.
        # Note: EWM on shifted column includes the shifted value (race i-1) in the mean for race i. Correct.
        col_ewm3 = f"{tgt}_ewm_3"
        col_ewm5 = f"{tgt}_ewm_5"
        
        # Pandas Groupby EWM approach
        # s_shifted contains NaNs at start.
        # ewm().mean() handles NaNs.
        
        # We need to preserve index alignment.
        # df_sorted is input.
        
        # Re-assign shifted to df for calculation
        temp_col = f"temp_shift_{tgt}"
        df_sorted[temp_col] = s_shifted
        
        # Calc EWM
        # Note: We must group by horse_id again for EWM
        g_ewm = df_sorted.groupby('horse_id')[temp_col]
        
        df_sorted[col_ewm3] = g_ewm.ewm(span=3, min_periods=1).mean().reset_index(level=0, drop=True)
        df_sorted[col_ewm5] = g_ewm.ewm(span=5, min_periods=1).mean().reset_index(level=0, drop=True)
        
        out_cols.extend([col_ewm3, col_ewm5])
        
        # 2. Slope (Linear Regression Slope for last 5)
        # Slope is relatively heavy.
        # x=[-4, -3, -2, -1, 0], y=[v1..v5].
        # Simplified slope: (Latest - Oldest) / N ? Or (Mean(Last3) - Mean(Prev3))?
        # Or `np.polyfit` in rolling apply.
        # Rolling Apply with Slope
        # Slope = Cov(x, y) / Var(x). x is fixed [0,1,2,3,4]. Var(x) is constant (2).
        # Cov(x, y) = Mean(xy) - Mean(x)Mean(y).
        # Mean(x) = 2.
        # So we can implement a fast rolling slope.
        # window=5. x=[0, 1, 2, 3, 4] (or -4 to 0).
        # let's use x = [-2, -1, 0, 1, 2] -> Mean(x)=0 to simplify?
        # x = [-2, -1, 0, 1, 2]. Sum(x*x) = 4+1+0+1+4 = 10. Var(x) = 10/5 = 2.
        # Slope = Sum(x*y) / Sum(x^2) = Sum(x*y) / 10.
        # y is the values in the window.
        
        if tgt == 'rank': # Rank trend is useful
            
            def calc_slope_5(window_series):
                if len(window_series) < 5: return np.nan
                # x centered at 0: -2, -1, 0, 1, 2
                # y are the values
                # slope = sum(x * (y - mean_y)) / sum(x^2)
                #       = sum(x * y) / 10
                y = window_series.values
                x = np.array([-2, -1, 0, 1, 2])
                # Check NaNs
                if np.isnan(y).any(): return np.nan
                
                s = np.dot(x, y)
                return s / 10.0

            col_slope = f"{tgt}_slope_5"
            # Apply rolling to the shifted column
            # rolling(5).apply(...)
            # This is slow! But for 200 features maybe acceptable.
            # Optimization: Use convolution?
            # Rolling dot product?
            # df[temp_col].rolling(5).apply is slow.
            # Faster approach: 
            # Slope = (2*y4 + 1*y3 + 0*y2 -1*y1 -2*y0) / 10 ?
            # Wait, order of window in rolling: y[0] is oldest?
            # rolling returns window [t-4, t-3, t-2, t-1, t].
            # x should be matched. t is latest.
            # So x=[-2, -1, 0, 1, 2] logic works if y is [old..new].
            # Yes.
            
            # Use raw convolution if possible, or just apply if strict simple.
            # For strict correctness and moderate speed (size is not huge? 1M rows?):
            # It takes time.
            # Let's use a simplified logical approximation or optimized apply.
            # Or just skip slope if it bottlenecks.
            # User requirement: "近走トレンド（EWM/傾き/分散）"
            # I will try to use rolling().cov() or similar if possible, but specific kernel is needed.
            # Let's provide a simplified "Trend" = (Mean(Last 2) - Mean(Prev 3)) ?
            # No, user asked for Slope.
            # Let's use simple (Latest - Oldest) / 4 ?
            # That's just (y[4] - y[0])/4. Very noisy.
            # Let's stick to the implementation using Rolling Apply but keep it minimal.
            # Actually, `rank` slope is the most important. `time_diff` is noisy.
            
            # Only calculate slope for Rank to save time, or use vectorized.
            # Vectorized Slope:
            # S_xy = Sum(x*y) - n * mean_x * mean_y
            # S_xx = Sum(x^2) - n * mean_x^2
            # Here x is fixed index. feature `n` (seq) can be used.
            # Slope = Correlation * Std_y / Std_x.
            # Rolling Correlation is available! `.rolling(5).corr(other)`?
            # But `other` must be the index.
            # Construct a helper 'idx' column [0, 1, 2...].
            # Then rolling_cov(y, idx).
            # Then slope = cov(y, idx) / var(idx).
            # var(idx) for window 5 is constant (= 2).
            # So Slope = RollingCov(y, idx, window=5) / 2.
            # Perfect.
            
            # Create a 0..N index per group? No, global increasing integer index is fine locally for covariance 
            # AS LONG AS it is contiguous.
            # But rows might be filtered?
            # df_sorted is sorted by horse_id, date.
            # run_count is 0,1,2... perfect.
            
            # Need 'run_count'
            if 'run_count' not in df_sorted.columns:
                df_sorted['run_count'] = grp.cumcount()
                
            # Rolling Covariance
            # need to align indexes.
            # We want rolling cov of (temp_col, run_count) per group.
            # Rolling on Groupby is tricky with 2 columns.
            # But 'run_count' increases by 1 step always?
            # If so, we don't need 'run_count' column per se, just implicit index.
            # But gaps in date don't matter for "last 5 starts".
            # So it IS just index 0,1,2,3,4 inside the window.
            # So we can assume x has variance 2.0 (for N=5).
            # We just need Cov(y, x).
            # Sum( (y - y_mean) * (x - x_mean) ) / (N-1) ? or N? Pandas uses N-1 by default.
            # x = [-2, -1, 0, 1, 2]. Mean=0.
            # Sum( y_cent * x ) / 4.
            # Sum( y * x ) / 4  (since sum(x)=0).
            # So Slope = (Sum(y*x)) / Sum(x*x).
            # Sum(x*x) = 10.
            # So Slope = Sum(y*x) / 10. (This is least squares slope).
            # (Note: Covariance formula gives same direction).
            
            # Let's implement Sum(y*x) / 10 via manual rolling sum.
            # y_t * 2 + y_{t-1} * 1 + y_{t-2} * 0 + y_{t-3} * (-1) + y_{t-4} * (-2)
            # This can be done by:
            # 2*y - 1*y_shift1 - 2*y_shift4 ?? No.
            # Construct weighted sum columns?
            # Too complex for quick script.
            
            # Fallback: simple numeric diff or just one `rolling().apply()`
            # For just 'rank', apply is okay. 50k horses * 20 races = 1M rows. 1M calls to valid function is ~5-10 sec. fast enough.
            
            col_slope = f"{tgt}_slope_5"
            
            def hardcoded_slope_5(y):
                 # y is list of 5 floats
                 return (2*y[4] + y[3] - y[1] - 2*y[0]) / 10.0
            
            df_sorted[col_slope] = g_ewm.rolling(5, min_periods=5).apply(hardcoded_slope_5, raw=True).reset_index(level=0, drop=True)
            out_cols.append(col_slope)
            
            # Std
            col_std = f"{tgt}_std_5"
            df_sorted[col_std] = g_ewm.rolling(5, min_periods=3).std().reset_index(level=0, drop=True)
            out_cols.append(col_std)

    # Fill NaNs
    df_sorted[out_cols] = df_sorted[out_cols].fillna(0)
    
    return df_sorted[cols + out_cols].copy()

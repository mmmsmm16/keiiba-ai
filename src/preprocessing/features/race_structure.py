import pandas as pd
import numpy as np

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 3.5] レース構造 (Race Structure)
    - 脚質分布、先行勢力
    - early_speed_dist: 逃げ先行馬の割合や総数
    - style_entropy: 脚質のばらつき
    """
    cols = ['race_id', 'horse_number', 'horse_id']
    df_sorted = df.copy()
    
    out_cols = []
    
    # Needs: 'pred_runstyle' (predicted runstyle) or 'last_nige_rate' or similar.
    # 'pred_runstyle' is created in 'runstyle_fit'.
    # If runstyle_fit is earlier, we can use it.
    # In registry, runstyle_fit is usually available.
    # If not, use 'last_nige_rate' from pace_stats.
    
    # Let's use 'last_nige_rate' (continuous) and 'last_3f' (proxy for late speed)
    
    if 'last_nige_rate' in df_sorted.columns:
        grp = df_sorted.groupby('race_id')
        
        # 1. Early Speed Sum (先行力総和)
        df_sorted['struct_early_speed_sum'] = grp['last_nige_rate'].transform('sum')
        df_sorted['struct_early_speed_mean'] = grp['last_nige_rate'].transform('mean')
        
        # 2. High Nige Count
        # [Fix] Lowered threshold from 0.5 to 0.2 - data shows max is ~0.4
        df_sorted['is_nige_cand'] = (df_sorted['last_nige_rate'] > 0.2).astype(int)
        df_sorted['struct_nige_count'] = grp['is_nige_cand'].transform('sum')
        
        out_cols.extend(['struct_early_speed_sum', 'struct_early_speed_mean', 'struct_nige_count'])
        
    # 3. Expected Pace (Regression logic or Proxy)
    # Higher early speed sum -> High Pace
    if 'struct_early_speed_sum' in df_sorted.columns:
        # Simple proxy: Just the sum itself is the feature "pace_proxy"
        # We can rename or alias
        df_sorted['pace_expectation_proxy'] = df_sorted['struct_early_speed_sum']
        out_cols.append('pace_expectation_proxy')
        
    # 4. Style Entropy
    # Use pred_runstyle if available (1:Nige, 2:Senko, 3:Sashi, 4:Oikomi)
    if 'pred_runstyle' in df_sorted.columns:
        # Calculate entropy of runstyles in race
        # P_i = count(style_i) / N
        # Entropy = -Sum(P_i * log(P_i))
        
        def calc_entropy(series):
            counts = series.value_counts([1,2,3,4], normalize=True) # normalize returns prob
            # Filter out zeros to avoid log(0)
            probs = counts[counts > 0]
            return -np.sum(probs * np.log(probs))
            
        # Group apply is slow. Transform with lambda?
        # Calculate counts per race
        # Pivot race_id x style -> counts
        # This is faster.
        
        pivot = pd.crosstab(df_sorted['race_id'], df_sorted['pred_runstyle'])
        # Normalize row-wise
        probs = pivot.div(pivot.sum(axis=1), axis=0)
        
        # Handle 0 for log
        import numpy as np
        # entropy = - sum (p * log(p))
        # Mask 0
        log_probs = np.log(probs.replace(0, 1.0)) # log(1)=0, so 0*0=0. Safe.
        entropy = -(probs * log_probs).sum(axis=1)
        
        # Map back
        df_sorted['style_entropy'] = df_sorted['race_id'].map(entropy).fillna(0)
        out_cols.append('style_entropy')

    return df_sorted[cols + out_cols].copy()

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

    # 5. Pace scenario probability (high-pace likelihood)
    if 'race_id' in df_sorted.columns:
        race_level = pd.DataFrame({'race_id': df_sorted['race_id'].unique()})
        race_level['field_size'] = df_sorted.groupby('race_id')['horse_number'].count().reindex(race_level['race_id']).values
        race_level['early_sum'] = df_sorted.groupby('race_id')['struct_early_speed_sum'].first().reindex(race_level['race_id']).fillna(0).values \
            if 'struct_early_speed_sum' in df_sorted.columns else 0.0
        race_level['nige_count'] = df_sorted.groupby('race_id')['struct_nige_count'].first().reindex(race_level['race_id']).fillna(0).values \
            if 'struct_nige_count' in df_sorted.columns else 0.0
        race_level['entropy'] = df_sorted.groupby('race_id')['style_entropy'].first().reindex(race_level['race_id']).fillna(0).values \
            if 'style_entropy' in df_sorted.columns else 0.0

        def zscore(s: pd.Series) -> pd.Series:
            std = s.std()
            if std is None or std == 0 or np.isnan(std):
                return pd.Series(np.zeros(len(s)), index=s.index)
            return (s - s.mean()) / std

        z_early = zscore(pd.to_numeric(race_level['early_sum'], errors='coerce').fillna(0.0))
        z_nige = zscore(pd.to_numeric(race_level['nige_count'], errors='coerce').fillna(0.0))
        z_entropy = zscore(pd.to_numeric(race_level['entropy'], errors='coerce').fillna(0.0))
        z_field = zscore(pd.to_numeric(race_level['field_size'], errors='coerce').fillna(0.0))

        # fixed coefficients for initial scenario model
        linear = -0.5 + 0.9 * z_early + 0.7 * z_nige + 0.4 * z_entropy + 0.3 * z_field
        race_level['pace_high_prob'] = 1.0 / (1.0 + np.exp(-linear))
        p_map = race_level.set_index('race_id')['pace_high_prob'].to_dict()
        df_sorted['pace_high_prob'] = df_sorted['race_id'].map(p_map).fillna(0.5)
        out_cols.append('pace_high_prob')

        if 'fit_sashi_high' in df_sorted.columns and 'fit_nige_slow' in df_sorted.columns:
            fit_high = pd.to_numeric(df_sorted['fit_sashi_high'], errors='coerce').fillna(0.0)
            fit_slow = pd.to_numeric(df_sorted['fit_nige_slow'], errors='coerce').fillna(0.0)
        elif 'pred_runstyle' in df_sorted.columns:
            fit_high = (df_sorted['pred_runstyle'] >= 3).astype(float)
            fit_slow = (df_sorted['pred_runstyle'] == 1).astype(float)
        else:
            fit_high = 0.0
            fit_slow = 0.0
        df_sorted['pace_fit_expected'] = df_sorted['pace_high_prob'] * fit_high + (1.0 - df_sorted['pace_high_prob']) * fit_slow
        out_cols.append('pace_fit_expected')

    # 6. Front congestion index
    if 'last_nige_rate' in df_sorted.columns:
        def calc_congestion(g: pd.DataFrame) -> pd.Series:
            n = len(g)
            if n <= 1:
                return pd.Series(np.zeros(n), index=g.index)
            p_front = np.clip(pd.to_numeric(g['last_nige_rate'], errors='coerce').fillna(0.0).values, 0.0, 1.0)
            if 'avg_first_corner_norm' in g.columns:
                mu = np.clip(pd.to_numeric(g['avg_first_corner_norm'], errors='coerce').fillna(0.5).values, 0.0, 1.0) * max(n - 1, 1) + 1.0
            else:
                mu = (1.0 - p_front) * max(n - 1, 1) + 1.0
            dist = np.abs(mu[:, None] - mu[None, :])
            w = np.exp(-dist / 1.5)
            np.fill_diagonal(w, 0.0)
            c = (w * p_front.reshape(1, -1)).sum(axis=1)
            return pd.Series(c, index=g.index)

        df_sorted['front_congestion_idx'] = df_sorted.groupby('race_id', group_keys=False).apply(calc_congestion)
        out_cols.append('front_congestion_idx')

    return df_sorted[cols + out_cols].copy()

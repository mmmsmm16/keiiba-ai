import pandas as pd
import numpy as np

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 3.4] レース条件 (Race Conditions)
    - going_code, weather_code: ローダーで取得したraw値をそのまま使用
    - track_variant: タイムの出方 (馬場差)
        - 安全設計: 「date < race_date」の過去レースのみを用いて基準タイムを算出し、乖離を見る
    """
    cols = ['race_id', 'horse_number', 'horse_id']
    df_sorted = df.sort_values('date').copy()
    
    out_cols = []
    
    # 1. Conditions
    if 'weather_code' in df_sorted.columns:
        out_cols.append('weather_code')
    if 'going_code' in df_sorted.columns:
        out_cols.append('going_code')
    
    # 2. Track Variant (馬場差)
    # logic:
    # 各開催(venue, distance, surface) ごとのタイムを監視。
    # 「直近50レース」の平均タイム - 「長期平均」 = Variant (マイナスなら高速馬場)
    # リーク防止: strictly past races only.
    
    if 'time' in df_sorted.columns and 'venue' in df_sorted.columns and 'distance' in df_sorted.columns:
        # Group by criteria
        # track_key: venue, surface, distance
        # Note: distance is numeric, track_code is surface
        
        # Calculate Rolling Mean of Win Times? Or All Times?
        # Usually Win Time (rank=1) is used for variant.
        # But sparse data.
        # Use All Times (Mean or Median of field) is more robust?
        # Or Top 3 Avg.
        # Let's use Top 3 Avg per race.
        
        # Agg per race first to avoid duplicate weighting by field size
        # Only rank<=3
        mask_top3 = (df_sorted['rank'] <= 3) & (df_sorted['time'] > 0)
        df_top3 = df_sorted[mask_top3].groupby(['race_id', 'date', 'venue', 'distance', 'surface'])['time'].mean().reset_index()
        df_top3.rename(columns={'time': 'race_avg_time'}, inplace=True)
        
        # Sort by date
        df_top3 = df_top3.sort_values('date')
        
        # Calculate Variant per group
        # keys: venue, surface, distance
        # We need to map back to race_id.
        
        # Use simple transform on sorted groups
        grp = df_top3.groupby(['venue', 'surface', 'distance'])
        
        # Rolling 10 mean (Past) - [Fix] Reduced from 50 to 10 for better JIT sensitivity
        # shift(1) is CRITICAL
        # [Fix] Reduced min_periods from 5 to 2 for sparse JIT data
        df_top3['rolling_time_10'] = grp['race_avg_time'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).mean())
        
        # Longterm Mean (Expanding)
        # shift(1) again
        # [Fix] Reduced min_periods from 5 to 2 for sparse JIT data
        df_top3['longterm_time'] = grp['race_avg_time'].transform(lambda x: x.shift(1).expanding(min_periods=2).mean())
        
        # [Fix] Broaden fallback: calculate variant per (venue, surface) to handle distance-specific sparsity
        grp_broad = df_top3.groupby(['venue', 'surface'])
        df_top3['broad_variant'] = grp_broad['race_avg_time'].transform(lambda x: x.shift(1).rolling(10, min_periods=2).mean()) - \
                                  grp_broad['race_avg_time'].transform(lambda x: x.shift(1).expanding(min_periods=2).mean())
        
        # Variant = Rolling - Longterm
        df_top3['track_variant'] = df_top3['rolling_time_10'] - df_top3['longterm_time']
        
        # [Fix] Apply fallback if null
        df_top3['track_variant'] = df_top3['track_variant'].fillna(df_top3['broad_variant']).fillna(0)
        
        # [Fix] For JIT prediction, we need to map the "last known" variant to the target race.
        # df_top3 has the chronological variants.
        # Create a mapping: (venue, surface, distance) -> latest track_variant
        latest_variants = {}
        for _, r in df_top3.sort_values('date').iterrows():
            key = (r['venue'], r['surface'], r['distance'])
            latest_variants[key] = r['track_variant']
            
        # Also a broader one for safety
        latest_broad = {}
        for _, r in df_top3.sort_values('date').iterrows():
            key = (r['venue'], r['surface'])
            latest_broad[key] = r['track_variant']

        # [Fix] Map back to df
        variant_map = df_top3.set_index('race_id')['track_variant'].to_dict()

        def get_variant(row):
            rid = row['race_id']
            if rid in variant_map:
                return variant_map[rid]
            
            # Fallback for target race (prediction)
            key = (row.get('venue'), row.get('surface'), row.get('distance'))
            if key in latest_variants:
                return latest_variants[key]
            
            key_b = (row.get('venue'), row.get('surface'))
            return latest_broad.get(key_b, 0)

        df_sorted['track_variant'] = df_sorted.apply(get_variant, axis=1)
        
        out_cols.append('track_variant')

    return df_sorted[cols + out_cols].copy()

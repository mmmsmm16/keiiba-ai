import pandas as pd
import numpy as np

def calculate_odds_movement_features(df_features, start_year=2014, end_year=2025, snapshot_dir='data/odds_snapshots'):
    """
    Calculate odds movement features for Phase 11.
    Features:
    - log_odds_t10: Log(Odds) at T-10m
    - dlog_odds_t60_to_t10: Log(T10) - Log(T60) (Trend)
    - dlog_odds_t30_to_t10: Log(T10) - Log(T30) (Short-term Trend)
    - volatility_t60_t10: StdDev of Log(Odds) across T60, T30, T10
    - rank_t10: Popularity Rank at T-10
    - rank_change_t60_to_t10: Rank(T10) - Rank(T60)
    
    Args:
        df_features: Base feature dataframe (must have race_id, horse_number)
        start_year, end_year: Range of years to load
    Returns:
        DataFrame with new features linked by [race_id, horse_number]
    """
    import os
    
    t60_list, t30_list, t10_list = [], [], []
    
    for year in range(start_year, end_year + 1):
        base_dir = f"{snapshot_dir}/{year}"
        if not os.path.exists(base_dir): 
            # print(f"Skipping {year}, no dir")
            continue
        
        # try:
        if True:
            # T-60
            p60 = f"{base_dir}/odds_T-60.parquet"
            if not os.path.exists(p60): continue
            
            df60 = pd.read_parquet(p60)
            df60 = df60[df60['ticket_type']=='win'].copy()
            df60['horse_number'] = pd.to_numeric(df60['combination'], errors='coerce').fillna(0).astype(int)
            df60 = df60[['race_id', 'horse_number', 'odds', 'ninki']]
            df60 = df60.rename(columns={'odds': 'odds_t60', 'ninki': 'ninki_t60'})
            t60_list.append(df60)

            # T-30
            df30 = pd.read_parquet(f"{base_dir}/odds_T-30.parquet")
            df30 = df30[df30['ticket_type']=='win'].copy()
            df30['horse_number'] = pd.to_numeric(df30['combination'], errors='coerce').fillna(0).astype(int)
            df30 = df30[['race_id', 'horse_number', 'odds', 'ninki']]
            df30 = df30.rename(columns={'odds': 'odds_t30', 'ninki': 'ninki_t30'})
            t30_list.append(df30)

            # T-10
            df10 = pd.read_parquet(f"{base_dir}/odds_T-10.parquet")
            df10 = df10[df10['ticket_type']=='win'].copy()
            df10['horse_number'] = pd.to_numeric(df10['combination'], errors='coerce').fillna(0).astype(int)
            df10 = df10[['race_id', 'horse_number', 'odds', 'ninki']]
            df10 = df10.rename(columns={'odds': 'odds_t10', 'ninki': 'ninki_t10'})
            t10_list.append(df10)
        # except Exception as e:
        #     print(f"Error loading {year}: {e}")
        #     pass
            
    if not t10_list:
        return pd.DataFrame()
        
    t60 = pd.concat(t60_list, ignore_index=True)
    t30 = pd.concat(t30_list, ignore_index=True)
    t10 = pd.concat(t10_list, ignore_index=True)
    
    # Merge
    # race_id, horse_number types
    t60['race_id'] = t60['race_id'].astype(str)
    t60['horse_number'] = t60['horse_number'].astype(int)
    t30['race_id'] = t30['race_id'].astype(str)
    t30['horse_number'] = t30['horse_number'].astype(int)
    t10['race_id'] = t10['race_id'].astype(str)
    t10['horse_number'] = t10['horse_number'].astype(int)
    
    combined = t10[['race_id', 'horse_number', 'odds_t10', 'ninki_t10']].copy()
    combined = pd.merge(combined, t30[['race_id', 'horse_number', 'odds_t30', 'ninki_t30']], on=['race_id', 'horse_number'], how='left')
    combined = pd.merge(combined, t60[['race_id', 'horse_number', 'odds_t60', 'ninki_t60']], on=['race_id', 'horse_number'], how='left')
    
    # Log Odds
    for c in ['odds_t60', 'odds_t30', 'odds_t10']:
        combined[f'log_{c}'] = np.log(combined[c].clip(1.0, 1000.0))
        
    # Features
    combined['dlog_odds_t60_t10'] = combined['log_odds_t10'] - combined['log_odds_t60']
    combined['dlog_odds_t30_t10'] = combined['log_odds_t10'] - combined['log_odds_t30']
    
    odds_cols = ['log_odds_t60', 'log_odds_t30', 'log_odds_t10']
    combined['odds_volatility'] = combined[odds_cols].std(axis=1)
    
    combined['rank_change_t60_t10'] = combined['ninki_t10'] - combined['ninki_t60']
    combined['odds_drop_rate_t60_t10'] = combined['odds_t10'] / combined['odds_t60']
    
    feature_cols = [
        'log_odds_t10',
        'dlog_odds_t60_t10',
        'dlog_odds_t30_t10',
        'odds_volatility',
        'rank_change_t60_t10',
        'odds_drop_rate_t60_t10'
    ]
    
    return combined[['race_id', 'horse_number'] + feature_cols]

import pandas as pd
import sys
import os
import numpy as np

# Adjust path to find src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing.loader import JraVanDataLoader
from src.preprocessing.feature_pipeline import FeaturePipeline

def main():
    print("Loading small sample...")
    loader = JraVanDataLoader()
    # Load 1 year
    df = loader.load(history_start_date='2024-01-01', skip_odds=True, skip_training=True)
    df = df.head(1000)
    print(f"Loaded {len(df)} rows.")
    
    pipeline = FeaturePipeline(cache_dir="data/features_t2/incremental_cache_debug")
    
    # 1. Debug Pace Stats Logic (Local)
    print("\n--- Pace Stats Logic Check ---")
    df_sorted = df.copy().sort_values(['horse_id', 'date'])
    
    def extract_first_corner(val):
        if pd.isna(val) or val == '': return np.nan
        try:
            parts = str(val).split('-')
            if len(parts) > 0: return float(parts[0])
        except: return np.nan
        return np.nan

    if 'passage_rate' in df_sorted.columns:
        df_sorted['first_corner_rank'] = df_sorted['passage_rate'].apply(extract_first_corner)
    elif 'pass_1' in df_sorted.columns:
         df_sorted['first_corner_rank'] = df_sorted['pass_1'].apply(extract_first_corner)
    else:
        df_sorted['first_corner_rank'] = np.nan
        
    print(f"first_corner_rank describe:\n{df_sorted['first_corner_rank'].describe()}")
    
    df_sorted['is_nige'] = (df_sorted['first_corner_rank'] == 1).astype(int)
    print(f"is_nige sum: {df_sorted['is_nige'].sum()}")
    
    grp = df_sorted.groupby('horse_id')
    df_sorted['last_nige_rate'] = grp['is_nige'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0)
    print(f"last_nige_rate describe:\n{df_sorted['last_nige_rate'].describe()}")
    
    # 2. Debug Sire Aptitude Logic (Local)
    print("\n--- Sire Aptitude Logic Check ---")
    df_blood = df.copy().sort_values(['sire_id', 'date'])
    df_blood['is_win'] = (df_blood['rank'] == 1).astype(int)
    
    heavy_codes = ['重', '不良', '03', '04', 3, 4, '3', '4']
    df_blood['is_heavy'] = df_blood['state'].isin(heavy_codes).astype(int)
    df_blood['win_heavy'] = df_blood['is_win'] & df_blood['is_heavy']
    
    print(f"is_heavy sum: {df_blood['is_heavy'].sum()}")
    print(f"win_heavy sum: {df_blood['win_heavy'].sum()}")
    
    grp_sire = df_blood.groupby('sire_id')
    
    # Check if cumsum works
    df_blood['sire_heavy_races'] = grp_sire['is_heavy'].transform(lambda x: x.cumsum().shift(1)).fillna(0)
    df_blood['sire_heavy_wins'] = grp_sire['win_heavy'].transform(lambda x: x.cumsum().shift(1)).fillna(0)
    
    print(f"sire_heavy_races describe:\n{df_blood['sire_heavy_races'].describe()}")
    print(f"sire_heavy_wins describe:\n{df_blood['sire_heavy_wins'].describe()}")
    
    def safe_div(a, b):
        return np.where(b > 0, a / b, 0)
        
    df_blood['sire_heavy_win_rate'] = safe_div(df_blood['sire_heavy_wins'], df_blood['sire_heavy_races'])
    print(f"sire_heavy_win_rate describe:\n{df_blood['sire_heavy_win_rate'].describe()}")


if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np

class HorseGearFeatures:
    """
    馬具（ブリンカーなど）に関する特徴量を生成する
    Properties:
    - has_blinker: ブリンカー着用の有無
    - is_first_blinker: 初ブリンカー（前走なし -> 今走あり）
    - is_blinker_off: ブリンカー外し（前走あり -> 今走なし）
    """
    
    def __init__(self):
        pass
        
    def fit(self, df: pd.DataFrame):
        pass
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Required columns check
        if 'blinker' not in df.columns:
            # If not present (e.g. not loaded), return empty or zeros
            # Usually loader ensures it exists if requested, but safe fallback
            return pd.DataFrame()
            
        df_sorted = df.sort_values(['horse_id', 'date']).copy()
        
        # 1. has_blinker (0 or 1)
        # Loader should have already converted it to int
        
        # 2. Previous Race Blinker
        grp = df_sorted.groupby('horse_id')
        df_sorted['lag1_blinker'] = grp['blinker'].shift(1).fillna(0)
        
        # 3. is_first_blinker
        # Current=1, Prev=0
        df_sorted['is_first_blinker'] = ((df_sorted['blinker'] == 1) & (df_sorted['lag1_blinker'] == 0)).astype(int)
        
        # 4. is_blinker_off
        # Current=0, Prev=1
        df_sorted['is_blinker_off'] = ((df_sorted['blinker'] == 0) & (df_sorted['lag1_blinker'] == 1)).astype(int)
        
        # Rename base column if needed or just keep 'blinker'
        # 'blinker' is already in df, but we want to return selected features.
        # If we return 'blinker', FeaturePipeline merge might duplicate or conflict if we are not careful.
        # FeaturePipeline merge uses left index (race_id, horse_number).
        # We should return the new columns + 'blinker' (renamed to 'has_blinker' for clarity if preferred, but 'blinker' is fine).
        
        # Rename for clarity
        df_sorted = df_sorted.rename(columns={'blinker': 'has_blinker'})
        
        features = ['has_blinker', 'is_first_blinker', 'is_blinker_off']
        keys = ['race_id', 'horse_number', 'horse_id']
        
        return df_sorted[keys + features].copy()

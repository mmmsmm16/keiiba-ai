import pandas as pd
import numpy as np

def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block 3.1] レース属性 (Race Attributes)
    - field_size: 出走頭数
    - handicap_type_code: ハンデ戦区分 (kyoso_shubetsu_code等から推定)
    """
    cols = ['race_id', 'horse_number', 'horse_id']
    if 'race_id' not in df.columns:
        return df[cols].copy() if set(cols).issubset(df.columns) else pd.DataFrame()

    df_sorted = df.copy()
    
    # 1. Field Size (出走頭数)
    # race_id ごとのカウント
    if 'race_id' in df_sorted.columns:
        # groupBy count
        race_counts = df_sorted.groupby('race_id')['horse_number'].count()
        df_sorted['field_size'] = df_sorted['race_id'].map(race_counts)
    else:
        df_sorted['field_size'] = np.nan

    # 2. Handicap Type / Allowance
    # kyoso_joken_code や kyoso_shubetsu_code, impost を使う
    # JRA data:
    # kyoso_shubetsu_code: 11,12,13...
    # impost: 斤量
    # ここでは簡易的に「ハンデ戦かどうか」のフラグなどを立てる
    # データ定義が不明確なため、kyoso_shubetsu_code をそのまま使うか、
    # 'impost_spread' (レース内の斤量バラつき) を特徴にする手が有効
    
    # Calculate impost spread in the race (std dev)
    if 'impost' in df_sorted.columns:
        impost_std = df_sorted.groupby('race_id')['impost'].std()
        df_sorted['race_impost_std'] = df_sorted['race_id'].map(impost_std).fillna(0)
        # バラつきが大きい(>0.5)ならハンデ戦の可能性が高い?
        df_sorted['is_handicap_race_guess'] = (df_sorted['race_impost_std'] > 0.8).astype(int)
    else:
        df_sorted['race_impost_std'] = 0.0
        df_sorted['is_handicap_race_guess'] = 0

    # Output columns
    out_cols = ['field_size', 'race_impost_std', 'is_handicap_race_guess']
    return df_sorted[cols + out_cols].copy()


import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_class_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Block] M4-B: Class Drift & Aptitude Stats
    - クラスレベルでの実績（昇級点・壁）を、最近の「クラス構造変化」に強くするために
      累積ではなく「直近1年」の実績として計算する。
    - クラス変動履歴（昇級直後か、降級か）も捉える。
    """
    logger.info("ブロック計算中: compute_class_stats (M4-B)")
    
    req_cols = ['horse_id', 'date', 'race_id', 'grade_code', 'rank', 'kyoso_joken_code']
    # grade_code or kyoso_joken_code determines class.
    # We should normalize class definition.
    # kyoso_joken_code: 005(1Win), 010(2Win), 016(3Win), 999(Open), etc.
    # grade_code: G1, G2, etc.
    
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        # Try to work with available? No, critical.
        raise ValueError(f"Missing columns for class_stats: {missing}")
        
    df_sorted = df.sort_values(['horse_id', 'date']).copy()
    if not np.issubdtype(df_sorted['date'].dtype, np.datetime64):
        df_sorted['date'] = pd.to_datetime(df_sorted['date'])
        
    # 1. Class Normalization (Labeling)
    # 簡易的に kyoso_joken_code を使用。
    # joken_code is string or int? usually string "005".
    # FillNa with 'Unknown'
    df_sorted['class_label'] = df_sorted['kyoso_joken_code'].fillna('999')
    
    # 2. Latest Class Experience (Rolling 365D)
    # horse_id x class_label ごとの集計
    # これは temporal_stats.compute_rolling_stats で「馬xクラス」をGroupにすればできる？
    # しかし「今走るクラス」での実績を取りたい。
    # A. 馬ごとの履歴を持ち、各行で「今回のクラス」と同じクラスの過去実績を集計する。
    #    -> transformだと全クラス混ざる。
    #    -> groupby(['horse_id', 'class_label']) で shift().rolling() して merge するのが王道。
    
    # Target
    df_sorted['is_top3'] = (df_sorted['rank'] <= 3).astype(int)
    
    # GroupBy Horse x Class
    grp_hc = df_sorted.groupby(['horse_id', 'class_label'])
    
    # Rolling 365D
    # groupby().rolling('365D', on='date')
    # closed='left'
    
    # Setup for rolling
    # Need to set index to date
    # But duplicates exist? (Same horse, same class, same date? - unlikely unless double header)
    
    temp = df_sorted.set_index('date')
    
    # Rolling
    # Performance optimization: filter only relevant classes? No, all history matters.
    r = temp.groupby(['horse_id', 'class_label'])['is_top3'].rolling('365D', closed='left')
    
    hc_stats = pd.DataFrame()
    hc_stats['hc_n_races_365d'] = r.count()
    hc_stats['hc_top3_sum_365d'] = r.sum()
    
    # Merge back
    # hc_stats index is (horse_id, class_label, date).
    # Reset index to clean up
    hc_stats = hc_stats.reset_index()
    
    # [Fix] Deduplicate hc_stats to prevent Cartesian product if index wasn't unique
    hc_stats = hc_stats.drop_duplicates(subset=['horse_id', 'class_label', 'date'])
    
    # Merge to df_sorted
    # Keys: horse_id, class_label, date
    df_sorted = pd.merge(df_sorted, hc_stats, on=['horse_id', 'class_label', 'date'], how='left')
    
    df_sorted['hc_n_races_365d'] = df_sorted['hc_n_races_365d'].fillna(0)
    df_sorted['hc_top3_rate_365d'] = (df_sorted['hc_top3_sum_365d'] / df_sorted['hc_n_races_365d']).fillna(0)
    
    # 3. Class Change Trend
    # 前走のクラス、前々走のクラス...
    grp_h = df_sorted.groupby('horse_id')
    df_sorted['prev_class'] = grp_h['class_label'].shift(1)
    df_sorted['prev2_class'] = grp_h['class_label'].shift(2)
    
    # Change Flag
    # 昇級判定は難しい（コードの大小?）。単純に「変化したか」
    # Same Class?
    df_sorted['is_same_class_prev'] = (df_sorted['class_label'] == df_sorted['prev_class']).astype(int)
    
    # Class Change Category (Experimental)
    # If code is numeric, we can check diff.
    # 005 < 010 < 016.
    def get_class_diff(row):
        try:
           curr = int(row['class_label'])
           prev = int(row['prev_class'])
           if curr > prev: return 1 # Promotion?
           if curr < prev: return -1 # Demotion?
           return 0
        except:
           return 0
           
    df_sorted['class_diff_val'] = df_sorted.apply(get_class_diff, axis=1)
    
    # Recent Trend
    # 過去3走の class_diff_val の平均 (昇級傾向にあるか、降級してきたか)
    # shift(1)してrolling
    df_sorted['class_trend_3'] = grp_h['class_diff_val'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
    
    # Features to keep
    feats = ['hc_n_races_365d', 'hc_top3_rate_365d', 'is_same_class_prev', 'class_trend_3']
    keys = ['race_id', 'horse_number', 'horse_id']
    
    return df_sorted[keys + feats].copy()
